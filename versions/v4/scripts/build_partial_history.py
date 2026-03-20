#!/usr/bin/env python3
"""
构建/同步精简版 stock_history（只存 ML picks 股票近 3 个月数据）

三种模式：
  1. build:  从 Polygon/Tushare 抓取 → 存本地 partial_stock_history.db
  2. upload: 本地 partial_stock_history.db → Supabase
  3. pull:   Supabase → 本地 stock_history.db (增量合并)

线上 Actions 流程:  build → upload
本地 cron 流程:      pull (从 Supabase 拉到本地 stock_history.db)

用法:
    PYTHONPATH=. python scripts/build_partial_history.py --market US         # build
    PYTHONPATH=. python scripts/build_partial_history.py --market US --upload # build + upload
    PYTHONPATH=. python scripts/build_partial_history.py --pull              # pull from Supabase
"""

import os, sys, sqlite3, argparse, time, json
from datetime import datetime, timedelta
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

DB_DIR = os.path.join(parent_dir, 'db')
PICKS_DB = os.path.join(DB_DIR, 'ml_daily_picks.db')
FULL_HIST_DB = os.path.join(DB_DIR, 'stock_history.db')
PARTIAL_DB = os.path.join(DB_DIR, 'partial_stock_history.db')

# ===================== Helpers =====================

def get_picks_symbols(market='US'):
    """Get unique symbols from ml_daily_picks.db"""
    if not os.path.exists(PICKS_DB):
        return []
    conn = sqlite3.connect(PICKS_DB)
    rows = conn.execute(
        'SELECT DISTINCT symbol FROM ml_picks_v2 WHERE market=?', (market,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def init_partial_db():
    """Initialize partial_stock_history.db"""
    conn = sqlite3.connect(PARTIAL_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS stock_history (
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, trade_date)
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ph_mkt_date ON stock_history(market, trade_date)')
    conn.commit()
    return conn


# ===================== Data Sources =====================

def copy_from_local(conn, symbols, market, start_date):
    """Copy from local full stock_history.db"""
    if not os.path.exists(FULL_HIST_DB):
        return 0
    full = sqlite3.connect(FULL_HIST_DB)
    ph = ','.join(['?'] * len(symbols))
    rows = full.execute(f'''
        SELECT symbol, market, trade_date, open, high, low, close, volume
        FROM stock_history WHERE symbol IN ({ph}) AND market=? AND trade_date>=?
    ''', list(symbols) + [market, start_date]).fetchall()
    full.close()
    if rows:
        conn.executemany(
            'INSERT OR REPLACE INTO stock_history VALUES (?,?,?,?,?,?,?,?)', rows)
        conn.commit()
    return len(rows)


def fetch_polygon(conn, symbols, market, start_date):
    """Fetch US from Polygon API (incremental)"""
    import urllib.request
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("  ⚠️ POLYGON_API_KEY not set")
        return 0
    total = 0
    end = datetime.now().strftime('%Y-%m-%d')
    for i, sym in enumerate(symbols):
        # Get last date we have
        last = conn.execute(
            'SELECT MAX(trade_date) FROM stock_history WHERE symbol=? AND market=?',
            (sym, market)).fetchone()[0]
        fetch_from = start_date
        if last and last >= start_date:
            next_day = (datetime.strptime(last, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            if next_day > end:
                continue
            fetch_from = next_day
        try:
            url = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
                   f"{fetch_from}/{end}?adjusted=true&sort=asc&limit=250&apiKey={api_key}")
            resp = urllib.request.urlopen(url, timeout=10)
            data = json.loads(resp.read())
            if data.get('results'):
                rows = [(sym, market,
                         datetime.fromtimestamp(r['t']/1000).strftime('%Y-%m-%d'),
                         r['o'], r['h'], r['l'], r['c'], r['v'])
                        for r in data['results']]
                conn.executemany(
                    'INSERT OR REPLACE INTO stock_history VALUES (?,?,?,?,?,?,?,?)', rows)
                total += len(rows)
            if (i+1) % 5 == 0:
                conn.commit()
            time.sleep(0.25)
        except Exception as e:
            if 'HTTP Error 429' in str(e):
                print(f"  ⚠️ Rate limited, sleeping 30s...")
                time.sleep(30)
            else:
                print(f"  ⚠️ {sym}: {e}")
    conn.commit()
    return total


def fetch_tushare(conn, symbols, market, start_date):
    """Fetch CN from Tushare"""
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("  ⚠️ TUSHARE_TOKEN not set")
        return 0
    try:
        import tushare as ts
        pro = ts.pro_api(token)
    except ImportError:
        print("  ⚠️ tushare not installed")
        return 0
    total = 0
    s_ts = start_date.replace('-', '')
    e_ts = datetime.now().strftime('%Y%m%d')
    for i, sym in enumerate(symbols):
        try:
            df = pro.daily(ts_code=sym, start_date=s_ts, end_date=e_ts)
            if df is not None and not df.empty:
                rows = [(sym, market,
                         f"{r['trade_date'][:4]}-{r['trade_date'][4:6]}-{r['trade_date'][6:8]}",
                         r['open'], r['high'], r['low'], r['close'], r['vol'])
                        for _, r in df.iterrows()]
                conn.executemany(
                    'INSERT OR REPLACE INTO stock_history VALUES (?,?,?,?,?,?,?,?)', rows)
                total += len(rows)
            if (i+1) % 10 == 0:
                conn.commit()
            time.sleep(0.15)
        except Exception as e:
            print(f"  ⚠️ {sym}: {e}")
    conn.commit()
    return total


# ===================== Supabase Sync =====================

def upload_to_supabase(market=None):
    """Upload partial_stock_history.db → Supabase table `partial_stock_history`"""
    try:
        from supabase import create_client
    except ImportError:
        print("  ⚠️ supabase not installed, skipping upload")
        return 0
    
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        print("  ⚠️ Supabase credentials not set")
        return 0
    
    sb = create_client(url, key)
    conn = sqlite3.connect(PARTIAL_DB)
    
    query = 'SELECT symbol, market, trade_date, open, high, low, close, volume FROM stock_history'
    params = []
    if market:
        query += ' WHERE market=?'
        params = [market]
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    if not rows:
        print("  ❌ No data to upload")
        return 0
    
    print(f"  📤 Uploading {len(rows)} rows to Supabase...")
    
    # Batch upsert (500 rows at a time for Supabase limits)
    batch_size = 500
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        records = [
            {'symbol': r[0], 'market': r[1], 'trade_date': r[2],
             'open': r[3], 'high': r[4], 'low': r[5], 'close': r[6], 'volume': r[7]}
            for r in batch
        ]
        try:
            sb.table('partial_stock_history').upsert(
                records, on_conflict='symbol,trade_date'
            ).execute()
            total += len(batch)
        except Exception as e:
            print(f"  ⚠️ Upsert batch error: {e}")
    
    print(f"  ✅ Uploaded {total} rows")
    return total


def pull_from_supabase(market=None, days=90):
    """Pull Supabase → local stock_history.db (merge)"""
    try:
        from supabase import create_client
    except ImportError:
        print("  ⚠️ supabase not installed")
        return 0
    
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        print("  ⚠️ Supabase credentials not set")
        return 0
    
    sb = create_client(url, key)
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Paginate fetch from Supabase
    all_rows = []
    offset = 0
    page_size = 1000
    while True:
        query = sb.table('partial_stock_history').select('*').gte('trade_date', start_date)
        if market:
            query = query.eq('market', market)
        query = query.range(offset, offset + page_size - 1)
        result = query.execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size
    
    if not all_rows:
        print("  ❌ No data from Supabase")
        return 0
    
    print(f"  📥 Got {len(all_rows)} rows from Supabase")
    
    # Merge into local stock_history.db
    local = sqlite3.connect(FULL_HIST_DB)
    local.execute('''
        CREATE TABLE IF NOT EXISTS stock_history (
            symbol TEXT NOT NULL, market TEXT NOT NULL, trade_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, market, trade_date)
        )
    ''')
    
    insert_rows = [
        (r['symbol'], r['market'], r['trade_date'],
         r.get('open'), r.get('high'), r.get('low'), r.get('close'), r.get('volume'))
        for r in all_rows
    ]
    local.executemany(
        'INSERT OR REPLACE INTO stock_history (symbol,market,trade_date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)',
        insert_rows)
    local.commit()
    local.close()
    
    print(f"  ✅ Merged {len(insert_rows)} rows into stock_history.db")
    return len(insert_rows)


# ===================== Main Build =====================

def build(market='US', days=90):
    """Build partial_stock_history.db for given market"""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    symbols = get_picks_symbols(market)
    if not symbols:
        print(f"  ❌ No {market} picks found")
        return
    print(f"\n📦 Build partial history: {market} ({len(symbols)} symbols, from {start_date})")
    
    conn = init_partial_db()
    
    # Try local full DB first
    n = copy_from_local(conn, symbols, market, start_date)
    if n:
        print(f"  ✅ {n} rows from local stock_history.db")
    
    # Fetch missing via API
    existing = set(r[0] for r in conn.execute(
        'SELECT DISTINCT symbol FROM stock_history WHERE market=? AND trade_date>=?',
        (market, start_date)).fetchall())
    missing = [s for s in symbols if s not in existing]
    
    if missing or True:  # Always try incremental update
        targets = missing if missing else symbols
        print(f"  📡 Fetching {len(targets)} symbols via API...")
        if market == 'US':
            n2 = fetch_polygon(conn, targets, market, start_date)
        else:
            n2 = fetch_tushare(conn, targets, market, start_date)
        if n2:
            print(f"  ✅ {n2} rows from API")
    
    # Cleanup old
    conn.execute('DELETE FROM stock_history WHERE trade_date < ?', (start_date,))
    conn.commit()
    conn.close()
    
    # VACUUM needs to run outside transaction
    vc = sqlite3.connect(PARTIAL_DB, isolation_level=None)
    vc.execute('VACUUM')
    vc.close()
    conn = sqlite3.connect(PARTIAL_DB)
    
    # Stats
    total = conn.execute('SELECT COUNT(*) FROM stock_history WHERE market=?', (market,)).fetchone()[0]
    syms = conn.execute('SELECT COUNT(DISTINCT symbol) FROM stock_history WHERE market=?', (market,)).fetchone()[0]
    conn.close()
    
    sz = os.path.getsize(PARTIAL_DB) / (1024*1024)
    print(f"  📊 {syms} symbols, {total} rows | DB size: {sz:.1f} MB")


# ===================== CLI =====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=['US', 'CN', 'BOTH'], default='BOTH')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--upload', action='store_true', help='Upload to Supabase after build')
    parser.add_argument('--pull', action='store_true', help='Pull from Supabase to local')
    args = parser.parse_args()
    
    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(parent_dir, '.env'))
    except ImportError:
        pass
    
    if args.pull:
        markets = ['US', 'CN'] if args.market == 'BOTH' else [args.market]
        for m in markets:
            print(f"\n📥 Pulling {m} from Supabase...")
            pull_from_supabase(m, args.days)
    else:
        markets = ['US', 'CN'] if args.market == 'BOTH' else [args.market]
        for m in markets:
            build(m, args.days)
        
        if args.upload:
            print(f"\n📤 Uploading to Supabase...")
            for m in markets:
                upload_to_supabase(m)
    
    print(f"\n✅ Done!")
