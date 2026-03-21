#!/usr/bin/env python3
"""
Pre-compute UI cache data to avoid slow API calls at page load time.

Caches:
  1. Market Pulse — index prices, BLUE signals, chip analysis, VIX, alt assets
  2. Market Caps — full A-share market caps from stock_history.db  
  3. Strategy NAV — pre-computed NAV curves from ml_daily_picks

Run as cron or after daily pipeline:
    python scripts/precompute_ui_cache.py
"""
import json
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

V4 = Path(__file__).resolve().parent.parent
DB_CACHE = V4 / 'db' / 'ui_cache.db'
DB_HIST  = V4 / 'db' / 'stock_history.db'
DB_PICKS = V4 / 'db' / 'ml_daily_picks.db'

import sys
sys.path.insert(0, str(V4))

# ── helpers ──────────────────────────────────────────────────────
def _init_cache_db():
    conn = sqlite3.connect(str(DB_CACHE))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _put(conn, key, obj):
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, value, updated_at) VALUES (?,?,?)",
        (key, json.dumps(obj, default=str), datetime.now().isoformat())
    )
    conn.commit()


# ── 1. Market Pulse ──────────────────────────────────────────────
def precompute_market_pulse(conn):
    """Compute index BLUE signals + chip for US & CN."""
    from indicator_utils import calculate_blue_signal_series
    from chart_utils import quick_chip_analysis
    from data_fetcher import get_cn_index_data

    for market in ('US', 'CN'):
        print(f"  [market_pulse] {market}...")
        result = {'_market': market}

        if market == 'CN':
            indices = {
                '000001.SH': {'name': '上证指数', 'emoji': '🔴'},
                '399001.SZ': {'name': '深证成指', 'emoji': '🟢'},
                '399006.SZ': {'name': '创业板指', 'emoji': '💡'},
                '000300.SH': {'name': '沪深300', 'emoji': '📊'},
            }
            fetcher = get_cn_index_data
            result['_currency'] = '¥'
        else:
            indices = {
                'SPY': {'name': 'S&P 500',      'emoji': '📊'},
                'QQQ': {'name': 'Nasdaq 100',    'emoji': '💻'},
                'DIA': {'name': 'Dow 30',        'emoji': '🏭'},
                'IWM': {'name': 'Russell 2000',  'emoji': '🏢'},
            }
            from data_fetcher import fetch_data_from_polygon
            fetcher = fetch_data_from_polygon
            result['_currency'] = '$'

        for symbol, info in indices.items():
            try:
                df_daily = fetcher(symbol, days=100)
                if df_daily is not None and len(df_daily) >= 30:
                    blue_daily = calculate_blue_signal_series(
                        df_daily['Open'].values, df_daily['High'].values,
                        df_daily['Low'].values, df_daily['Close'].values
                    )
                    df_weekly = df_daily.resample('W-MON').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min',
                        'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                    blue_weekly = [0]
                    if len(df_weekly) >= 10:
                        blue_weekly = calculate_blue_signal_series(
                            df_weekly['Open'].values, df_weekly['High'].values,
                            df_weekly['Low'].values, df_weekly['Close'].values
                        )
                    chip_result = quick_chip_analysis(df_daily)
                    chip_pattern = chip_result.get('label', '') if chip_result else ''

                    latest_price = float(df_daily['Close'].iloc[-1])
                    prev_price = float(df_daily['Close'].iloc[-2]) if len(df_daily) > 1 else latest_price
                    price_change = (latest_price - prev_price) / prev_price * 100

                    result[symbol] = {
                        'name': info['name'], 'emoji': info['emoji'],
                        'price': latest_price, 'change': round(price_change, 2),
                        'day_blue': float(blue_daily[-1]) if len(blue_daily) > 0 else 0,
                        'week_blue': float(blue_weekly[-1]) if len(blue_weekly) > 0 else 0,
                        'chip': chip_pattern,
                    }
                else:
                    result[symbol] = {'name': info['name'], 'emoji': info['emoji'],
                                      'price': 0, 'change': 0, 'day_blue': 0,
                                      'week_blue': 0, 'chip': ''}
            except Exception as e:
                print(f"    ⚠ {symbol}: {e}")
                result[symbol] = {'name': info['name'], 'emoji': info['emoji'],
                                  'price': 0, 'change': 0, 'day_blue': 0,
                                  'week_blue': 0, 'chip': '', 'error': str(e)}

        # VIX (US only)
        if market == 'US':
            try:
                vix_df = fetch_data_from_polygon('VIXY', days=30)
                if vix_df is not None and len(vix_df) > 0:
                    vix_price = float(vix_df['Close'].iloc[-1])
                    vix_prev = float(vix_df['Close'].iloc[-2]) if len(vix_df) > 1 else vix_price
                    vix_change = vix_price - vix_prev
                    if vix_price < 20:    vix_mood = "😌 极度贪婪"
                    elif vix_price < 25:  vix_mood = "🙂 平静"
                    elif vix_price < 30:  vix_mood = "😐 中性"
                    elif vix_price < 40:  vix_mood = "😟 焦虑"
                    else:                 vix_mood = "😱 恐惧"
                    result['VIX'] = {'price': vix_price, 'change': round(vix_change, 2), 'mood': vix_mood}
                else:
                    result['VIX'] = {'price': 0, 'change': 0, 'mood': '数据不可用'}
            except:
                result['VIX'] = {'price': 0, 'change': 0, 'mood': '未知'}

            # Alt assets (Gold, Silver, BTC)
            alt_assets = {
                'GLD': {'name': '黄金', 'emoji': '🥇', 'format': '${:.2f}'},
                'SLV': {'name': '白银', 'emoji': '🥈', 'format': '${:.2f}'},
                'X:BTCUSD': {'name': 'BTC', 'emoji': '₿', 'format': '${:,.0f}'},
            }
            for sym, ainfo in alt_assets.items():
                try:
                    adf = fetch_data_from_polygon(sym, days=30)
                    if adf is not None and len(adf) > 0:
                        price = float(adf['Close'].iloc[-1])
                        prev = float(adf['Close'].iloc[-2]) if len(adf) > 1 else price
                        chg = (price - prev) / prev * 100
                        result[sym] = {'name': ainfo['name'], 'emoji': ainfo['emoji'],
                                       'price': price, 'change': round(chg, 2),
                                       'format': ainfo['format']}
                except:
                    result[sym] = {'name': ainfo['name'], 'emoji': ainfo['emoji'],
                                   'price': 0, 'change': 0, 'format': ainfo['format']}

        # Sentiment
        main_indices = [k for k in result.keys() if not k.startswith('_') and k not in ['VIX', 'GLD', 'SLV', 'X:BTCUSD']]
        bullish_count = sum(1 for k in main_indices if result.get(k, {}).get('day_blue', 0) > 100)
        vix_ok = result.get('VIX', {}).get('price', 20) < 25 if market == 'US' else True

        if bullish_count >= 3 and vix_ok:
            sentiment = ("🟢 强势做多", "进攻型 60-80%", "#3fb950")
        elif bullish_count >= 2:
            sentiment = ("🟡 震荡偏多", "平衡型 40-60%", "#d29922")
        elif bullish_count >= 1:
            sentiment = ("🟠 分化观望", "防守型 20-40%", "#f85149")
        else:
            sentiment = ("🔴 弱势防守", "空仓或对冲", "#f85149")

        result['_sentiment'] = sentiment
        result['_bullish_count'] = bullish_count

        _put(conn, f"market_pulse_{market}", result)
        print(f"  ✅ market_pulse_{market}: {len(main_indices)} indices")


# ── 2. Market Caps ───────────────────────────────────────────────
def precompute_market_caps(conn):
    """Cache market caps from stock_history.db for instant lookup."""
    print("  [market_caps]...")
    if not DB_HIST.exists():
        print("    ⚠ stock_history.db not found, skipping")
        return

    hconn = sqlite3.connect(str(DB_HIST))
    
    # CN: get latest market caps from stock_history 
    try:
        cn_caps = {}
        # Try to get market_cap from stock_history if available
        cols = [r[1] for r in hconn.execute("PRAGMA table_info(stock_history)").fetchall()]
        if 'total_mv' in cols:
            rows = hconn.execute("""
                SELECT symbol, total_mv FROM stock_history 
                WHERE market='CN' AND total_mv > 0
                AND trade_date = (SELECT MAX(trade_date) FROM stock_history WHERE market='CN')
            """).fetchall()
            cn_caps = {r[0]: r[1] / 1e8 for r in rows}  # 转为亿
        
        if not cn_caps:
            # Fallback: try AkShare
            try:
                import akshare as ak
                spot_df = ak.stock_zh_a_spot_em()
                cn_caps = dict(zip(spot_df['代码'], spot_df['总市值']))
            except Exception as e:
                print(f"    ⚠ AkShare fallback failed: {e}")
        
        if cn_caps:
            _put(conn, "market_caps_CN", cn_caps)
            print(f"  ✅ market_caps_CN: {len(cn_caps)} stocks")
    except Exception as e:
        print(f"    ⚠ CN market caps: {e}")

    # US: get from stock_history if possible
    try:
        us_caps = {}
        cols = [r[1] for r in hconn.execute("PRAGMA table_info(stock_history)").fetchall()]
        if 'market_cap' in cols:
            rows = hconn.execute("""
                SELECT symbol, market_cap FROM stock_history
                WHERE market='US' AND market_cap > 0
                AND trade_date = (SELECT MAX(trade_date) FROM stock_history WHERE market='US')
            """).fetchall()
            us_caps = {r[0]: r[1] / 1e9 for r in rows}  # 转为 billions
        
        if us_caps:
            _put(conn, "market_caps_US", us_caps)
            print(f"  ✅ market_caps_US: {len(us_caps)} stocks")
    except Exception as e:
        print(f"    ⚠ US market caps: {e}")

    hconn.close()


# ── 3. Strategy NAV ──────────────────────────────────────────────
def precompute_strategy_navs(conn):
    """Pre-compute strategy NAV curves from ml_daily_picks."""
    print("  [strategy_navs]...")
    if not DB_PICKS.exists():
        print("    ⚠ ml_daily_picks.db not found, skipping")
        return

    pconn = sqlite3.connect(str(DB_PICKS))

    STRAT_DEFS = [
        ('MID_10',   ['Mid ($2-10B)'],  0.10, 20, 1),
        ('LARGE_10', ['Large ($10-100B)'], 0.10, 20, 1),
        ('SMALL_10', ['Small ($300M-2B)'], 0.10, 20, 1),
        ('ALL_3PCT', ['Large ($10-100B)', 'Mid ($2-10B)', 'Small ($300M-2B)'], 0.03, 20, 1),
        ('MID_TOP3', ['Mid ($2-10B)'],  0.03, 20, 3),
        ('MS_5DAY',  ['Mid ($2-10B)', 'Small ($300M-2B)'], 0.10, 5, 1),
    ]

    for market in ('US', 'CN'):
        df = pd.read_sql(
            "SELECT * FROM mmoe_daily_picks WHERE market=? ORDER BY date",
            pconn, params=(market,)
        )
        if df.empty:
            continue

        trade_dates = sorted(df['date'].unique())
        result = {}

        for sname, tiers, pct, hold, top_n in STRAT_DEFS:
            initial_cap = 100.0
            cumulative_pl = 0.0
            curve = []
            rets = []
            tier_df = df[df['tier'].isin(tiers)]

            for dt in trade_dates:
                day = tier_df[tier_df['date'] == dt].head(top_n)
                if not day.empty:
                    ret_col = f'actual_{hold}d' if f'actual_{hold}d' in day.columns else 'actual_20d'
                    day_rets = day[ret_col].dropna()
                    if not day_rets.empty:
                        for r in day_rets.tolist():
                            r_capped = max(-100, min(100, r))
                            position_size = initial_cap * pct
                            cumulative_pl += position_size * r_capped / 100
                            rets.append(r_capped)

                nav = initial_cap + cumulative_pl
                curve.append({'date': dt, 'nav': round(nav, 2)})

            if rets:
                wins = sum(1 for r in rets if r > 0)
                navs = [c['nav'] for c in curve]
                peak = navs[0]
                mdd = 0
                for n in navs:
                    if n > peak: peak = n
                    dd = (peak - n) / peak * 100
                    if dd > mdd: mdd = dd
                result[sname] = {
                    'curve': curve,
                    'stats': {
                        'wr': round(wins / len(rets) * 100, 1),
                        'avg': round(float(np.mean(rets)), 1),
                        'mdd': round(mdd, 1),
                        'n': len(rets),
                        'nav': round(nav, 1),
                    }
                }
            else:
                result[sname] = {
                    'curve': curve,
                    'stats': {'wr': 0, 'avg': 0, 'mdd': 0, 'n': 0, 'nav': 100}
                }

        _put(conn, f"strategy_navs_{market}", result)
        print(f"  ✅ strategy_navs_{market}: {len(result)} strategies")

    pconn.close()


# ── Main ─────────────────────────────────────────────────────────
def main():
    print(f"🚀 Pre-computing UI cache... ({datetime.now().isoformat()})")
    t0 = time.time()

    conn = _init_cache_db()

    try:
        precompute_market_pulse(conn)
    except Exception as e:
        print(f"❌ market_pulse failed: {e}")
        traceback.print_exc()

    try:
        precompute_market_caps(conn)
    except Exception as e:
        print(f"❌ market_caps failed: {e}")
        traceback.print_exc()

    try:
        precompute_strategy_navs(conn)
    except Exception as e:
        print(f"❌ strategy_navs failed: {e}")
        traceback.print_exc()

    conn.close()
    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.1f}s. Cache: {DB_CACHE}")


if __name__ == '__main__':
    main()
