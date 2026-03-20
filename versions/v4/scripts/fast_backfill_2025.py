import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import get_connection, insert_scan_result
from data_fetcher import get_all_us_tickers
from ml.data_cache import DataCache
from indicator_utils import (
    calculate_blue_signal_series, calculate_heima_signal_series,
    calculate_adx_series, calculate_volume_profile_metrics,
    calculate_volatility, analyze_elliott_wave_proxy, analyze_chanlun_proxy
)

def _ema(vals, n):
    alpha = 2.0 / (n + 1.0)
    out = np.zeros(len(vals))
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i-1]
    return out

def _sma_cn(vals, n, m=1):
    out = np.zeros(len(vals))
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = (m * vals[i] + (n - m) * out[i-1]) / n
    return out

def process_symbol(symbol, start_date, end_date):
    cache = DataCache()
    df = cache.get_stock_history(symbol, market='US', days=3650, force_refresh=False)
    if df is None or len(df) < 60:
        return []
    
    # ensure column names
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    if 'date' in df.columns:
        df = df.set_index('date')
        
    df = df.sort_index()
    
    # Pre-calculate indicator
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    volumes = df['Volume'].values
    
    day_blue = calculate_blue_signal_series(opens, highs, lows, closes)
    heima_arr, juedi_arr = calculate_heima_signal_series(highs, lows, closes, opens)
    adx_arr = calculate_adx_series(highs, lows, closes, period=14)
    volatility_arr = pd.Series(closes).pct_change().rolling(252).std().values * np.sqrt(252)
    
    # Weekly Series
    df_weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    valid_weekly = ~df_weekly['Close'].isna()
    df_weekly = df_weekly[valid_weekly]
    if len(df_weekly) >= 10:
        wk_open, wk_high, wk_low, wk_close = df_weekly['Open'].values, df_weekly['High'].values, df_weekly['Low'].values, df_weekly['Close'].values
        week_blue = calculate_blue_signal_series(wk_open, wk_high, wk_low, wk_close)
        heima_w, juedi_w = calculate_heima_signal_series(wk_high, wk_low, wk_close, wk_open)
        df_weekly['week_blue'] = week_blue
        df_weekly['heima_w'] = heima_w
        df_weekly['juedi_w'] = juedi_w
    else:
        df_weekly['week_blue'] = 0
        df_weekly['heima_w'] = False
        df_weekly['juedi_w'] = False
        
    # Monthly Series
    df_monthly = df.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    valid_monthly = ~df_monthly['Close'].isna()
    df_monthly = df_monthly[valid_monthly]
    if len(df_monthly) >= 6:
        m_open, m_high, m_low, m_close = df_monthly['Open'].values, df_monthly['High'].values, df_monthly['Low'].values, df_monthly['Close'].values
        month_blue = calculate_blue_signal_series(m_open, m_high, m_low, m_close)
        heima_m, juedi_m = calculate_heima_signal_series(m_high, m_low, m_close, m_open)
        df_monthly['month_blue'] = month_blue
        df_monthly['heima_m'] = heima_m
        df_monthly['juedi_m'] = juedi_m
    else:
        df_monthly['month_blue'] = 0
        df_monthly['heima_m'] = False
        df_monthly['juedi_m'] = False

    # Duokongwang
    n = len(closes)
    dkw_buy = np.zeros(n, dtype=bool)
    dkw_sell = np.zeros(n, dtype=bool)
    try:
        if n >= 30:
            opens_proxy = np.pad(closes[:-1], (1, 0), constant_values=closes[0])
            up = _ema(highs, 13)
            dw = _ema(lows, 13)
            
            # KDJ
            rsv = np.zeros(n)
            for i in range(13, n):
                llv = np.min(lows[max(0, i-13):i+1])
                hhv = np.max(highs[max(0, i-13):i+1])
                rsv[i] = 50.0 if hhv<=llv else (closes[i]-llv)/(hhv-llv)*100.0
            rsv[:13] = 50.0
            k = _sma_cn(rsv, 3, 1)
            d = _sma_cn(k, 3, 1)
            j = 3.0*k - 2.0*d
            
            # rsi
            lc = np.pad(closes[:-1], (1, 0), constant_values=closes[0])
            up_move = np.maximum(closes-lc, 0)
            abs_move = np.abs(closes-lc)
            rsi_num = _sma_cn(up_move, 9, 1)
            rsi_den = _sma_cn(abs_move, 9, 1)
            rsi2 = np.where(rsi_den>1e-12, rsi_num/rsi_den*100.0, 50.0)
            
            # nine turn
            nt = np.zeros(n, dtype=int)
            nt0 = np.zeros(n, dtype=int)
            a1 = np.zeros(n, dtype=bool)
            a1[4:] = closes[4:] > closes[:-4]
            b1 = np.zeros(n, dtype=bool)
            b1[4:] = closes[4:] < closes[:-4]
            for i in range(1, n):
                if a1[i]: nt[i] = nt[i-1] + 1
                if b1[i]: nt0[i] = nt0[i-1] + 1
    
            for i in range(20, n):
                cond = ((closes[i] > opens_proxy[i] and (opens_proxy[i] > up[i] or closes[i] < dw[i])) or 
                        (closes[i] < opens_proxy[i] and (opens_proxy[i] < dw[i] or closes[i] > up[i])))
                cond1 = up[i] > up[i-1] and dw[i] > dw[i-1]
                cond2 = up[i] < up[i-1] and dw[i] < dw[i-1]
                
                j_cross_level, j_oversold_prev = 30.0, 22.0
                rsi_prev_th, rsi_now_th, nine_min = 24.0, 20.0, 9
                
                kdj_cross_up = j[i-1]<=j_cross_level and j[i]>j_cross_level
                kdj_oversold_turn = j[i-1]<j_oversold_prev and j[i]>j[i-1]
                rsi_oversold_turn = rsi2[i-1]<=rsi_prev_th and rsi2[i]>rsi_now_th
                nine_down_exhaust = nt0[i]>=nine_min
                dkw_buy[i] = (cond and cond1 and (kdj_cross_up or rsi_oversold_turn)) or kdj_oversold_turn or rsi_oversold_turn or nine_down_exhaust
                
                kdj_overheat_fade = (j[i-1]>=100.0 and j[i]<95.0) or (j[i-1]>=90.0 and j[i]<j[i-1]-8.0)
                rsi_overbought_turn = rsi2[i-1]>=79.0 and rsi2[i]<80.0
                nine_up_exhaust = nt[i]>=9 and closes[i]<closes[i-1]
                dkw_sell[i] = (cond and cond2) or kdj_overheat_fade or rsi_overbought_turn or nine_up_exhaust
    except Exception:
        pass
        
    results = []
    
    # Now find target dates in 2025
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except:
        pass
        
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date).replace(hour=23, minute=59, second=59)
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    target_indices = np.where(mask)[0]
    
    df_wk_idx = df_weekly.index.tz_localize(None)
    df_m_idx = df_monthly.index.tz_localize(None)
    
    for i in target_indices:
        if i < 60: continue
        
        d_blue = float(day_blue[i])
        d_heima = bool(heima_arr[i])
        d_juedi = bool(juedi_arr[i])
        
        dt_naive = df.index[i].tz_localize(None)
        
        # approximate weekly and monthly indexing
        # weekly: first Friday >= dt_naive
        # simply use forward fill logic: closest past/current period
        w_mask = df_wk_idx <= dt_naive
        w_blue, w_heima, w_juedi = 0.0, False, False
        if np.any(w_mask):
            last_w = df_weekly.iloc[np.where(w_mask)[0][-1]]
            w_blue = float(last_w['week_blue'])
            w_heima = bool(last_w['heima_w'])
            w_juedi = bool(last_w['juedi_w'])
            
        m_mask = df_m_idx <= dt_naive
        m_blue, m_heima, m_juedi = 0.0, False, False
        if np.any(m_mask):
            last_m = df_monthly.iloc[np.where(m_mask)[0][-1]]
            m_blue = float(last_m['month_blue'])
            m_heima = bool(last_m['heima_m'])
            m_juedi = bool(last_m['juedi_m'])
            
        is_strat_d = d_blue >= 80
        is_strat_c = (d_blue >= 64) and (d_heima or d_juedi or w_blue > 100)
        legacy = d_blue >= 100
        
        has_signal = is_strat_d or is_strat_c or legacy or w_blue > 100 or dkw_buy[i]
        
        if not has_signal: continue
        
        # calculate remaining heavy indicators only for matches
        sub_closes = closes[:i+1]
        sub_highs = highs[:i+1]
        sub_lows = lows[:i+1]
        sub_vols = volumes[:i+1]
        curr_price = closes[i]
        
        try:
            vp_res = calculate_volume_profile_metrics(sub_closes, sub_vols, curr_price)
            vp_rating = "Normal"
            if vp_res['profit_ratio'] > 0.9: vp_rating = "Excellent"
            elif vp_res['profit_ratio'] > 0.7: vp_rating = "Good"
            elif vp_res['profit_ratio'] < 0.3: vp_rating = "Poor"
        except:
            vp_res = None
            vp_rating = "Normal"
            
        wave_res = analyze_elliott_wave_proxy(sub_closes, sub_highs, sub_lows)
        wave_phase = wave_res.get('phase', '') if isinstance(wave_res, dict) else ''
        wave_desc = wave_res.get('desc', '') if isinstance(wave_res, dict) else ''
        
        chan_res = analyze_chanlun_proxy(sub_closes, sub_highs, sub_lows)
        chan_signal = chan_res.get('signal', '') if isinstance(chan_res, dict) else ''
        chan_desc = chan_res.get('desc', '') if isinstance(chan_res, dict) else ''
        
        volatility = volatility_arr[i] if not np.isnan(volatility_arr[i]) else 0
        adx_val = adx_arr[i] if not np.isnan(adx_arr[i]) else 0
        
        regime = "Standard"
        if volatility > 0.6: regime = "High Vol (妖股)"
        elif volatility < 0.25: regime = "Mid-Low Vol"
        if adx_val > 30: regime += " | 强趋势"
        
        turnover = (curr_price * sub_vols[-1]) / 1000000 if len(sub_vols)>0 else 0
        
        stop_loss = curr_price * 0.95
        risk_per_share = curr_price - stop_loss
        shares_rec = int(1000 / risk_per_share) if risk_per_share > 0 else 0
        
        results.append({
            'Symbol': symbol,
            'Date': dt_naive.strftime('%Y-%m-%d'),
            'ML_Rank_Score': None,
            'Price': round(curr_price, 2),
            'Turnover_M': round(turnover, 2),
            'Blue_Daily': round(d_blue, 1),
            'Blue_Weekly': round(w_blue, 1),
            'Blue_Monthly': round(m_blue, 1),
            'Day_High': round(highs[i], 4),
            'Day_Low': round(lows[i], 4),
            'Day_Close': round(closes[i], 4),
            'ADX': round(adx_val, 1),
            'Volatility': round(volatility, 2),
            'Is_Heima': bool(d_heima or w_heima or m_heima),
            'Is_Juedi': bool(d_juedi or w_juedi or m_juedi),
            'Heima_Daily': bool(d_heima),
            'Heima_Weekly': bool(w_heima),
            'Heima_Monthly': bool(m_heima),
            'Juedi_Daily': bool(d_juedi),
            'Juedi_Weekly': bool(w_juedi),
            'Juedi_Monthly': bool(m_juedi),
            'Strat_D_Trend': is_strat_d,
            'Strat_C_Resonance': is_strat_c,
            'Legacy_Signal': legacy,
            'Regime': regime,
            'Adaptive_Thresh': 80,
            'VP_Rating': vp_rating,
            'Profit_Ratio': round(vp_res['profit_ratio'], 4) if vp_res else 0,
            'Wave_Phase': wave_phase,
            'Wave_Desc': wave_desc,
            'Chan_Signal': str(chan_signal),
            'Chan_Desc': chan_desc,
            'Duokongwang_Buy': bool(dkw_buy[i]),
            'Duokongwang_Sell': bool(dkw_sell[i]),
            'Market_Cap': None,
            'Cap_Category': 'Unknown',
            'Company_Name': None,
            'Industry': None,
            'Stop_Loss': stop_loss,
            'Shares_Rec': shares_rec,
            'Risk_Reward_Score': round(vp_res['profit_ratio']*100 if vp_res else adx_val, 2),
            'Market': 'US'
        })
        
    return results

def run_fast_backfill(start_date, end_date):
    tickers = get_all_us_tickers()
    all_results = []
    print(f"Fast backfilling {len(tickers)} tickers from {start_date} to {end_date}")
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(process_symbol, t, start_date, end_date): t for t in tickers}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = future.result()
                if res:
                    # we insert them in batch
                    for r in res:
                        all_results.append(r)
            except Exception as e:
                import traceback
                print(f"Error process: {e}")
                traceback.print_exc()
                
    print(f"Total signals found: {len(all_results)}. Inserting to DB...")
    # chunk inserts
    conn = get_connection()
    cursor = conn.cursor()
    
    chunk_size = 5000
    for i in range(0, len(all_results), chunk_size):
        chunk = all_results[i:i+chunk_size]
        for r in chunk:
            insert_scan_result(r)
            
    conn.commit()
    conn.close()
    print("Done")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    args = parser.parse_args()
    run_fast_backfill(args.start, args.end)
