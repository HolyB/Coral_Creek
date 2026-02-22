"""
Kronos 预计算脚本
==================
独立于 Streamlit 运行，提前为指定股票批量计算 Kronos 预测结果，
将结果缓存到 SQLite 数据库，网页端直接读取即可秒开。

用法:
    # 预测指定股票
    python scripts/kronos_precompute.py HIMS AAPL NVDA TSLA

    # 预测今日扫描出的所有 BLUE 信号股票
    python scripts/kronos_precompute.py --from-signals

    # 预测全部 (从 signals 中 + 自选列表)
    python scripts/kronos_precompute.py --from-signals HIMS AAPL
"""
import os
import sys
import json
import sqlite3
import argparse
import time
from datetime import datetime

# 确保项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(SCRIPT_DIR)
if V3_DIR not in sys.path:
    sys.path.insert(0, V3_DIR)

from ml.kronos_integration import get_kronos_engine
from ml.data_cache import DataCache

# 预测结果缓存数据库
CACHE_DB = os.path.join(V3_DIR, "db", "kronos_cache.db")


def init_cache_db():
    """初始化缓存数据库表"""
    os.makedirs(os.path.dirname(CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kronos_predictions (
            symbol TEXT NOT NULL,
            market TEXT NOT NULL DEFAULT 'US',
            pred_date TEXT NOT NULL,
            pred_len INTEGER NOT NULL DEFAULT 20,
            temperature REAL NOT NULL DEFAULT 0.5,
            prediction_json TEXT NOT NULL,
            last_hist_date TEXT,
            last_hist_close REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (symbol, market, pred_date, pred_len)
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(symbol: str, market: str, pred_df, last_hist_date: str, last_hist_close: float,
                    pred_len: int = 20, temperature: float = 0.5):
    """保存预测结果到缓存数据库"""
    conn = sqlite3.connect(CACHE_DB)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 将 DataFrame 转成 JSON
    pred_records = []
    for idx, row in pred_df.iterrows():
        pred_records.append({
            "date": str(idx)[:10],
            "Open": round(float(row["Open"]), 4),
            "High": round(float(row["High"]), 4),
            "Low": round(float(row["Low"]), 4),
            "Close": round(float(row["Close"]), 4),
            "Volume": round(float(row["Volume"]), 2),
        })
    
    conn.execute("""
        INSERT OR REPLACE INTO kronos_predictions 
        (symbol, market, pred_date, pred_len, temperature, prediction_json, last_hist_date, last_hist_close, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, market, today, pred_len, temperature, json.dumps(pred_records),
          last_hist_date, last_hist_close, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def load_prediction(symbol: str, market: str = "US", pred_date: str = None):
    """从缓存数据库读取预测结果"""
    import pandas as pd
    if not os.path.exists(CACHE_DB):
        return None
    if pred_date is None:
        pred_date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(CACHE_DB)
    row = conn.execute("""
        SELECT prediction_json, last_hist_date, last_hist_close, temperature, pred_len, created_at
        FROM kronos_predictions 
        WHERE symbol=? AND market=? AND pred_date=?
        ORDER BY created_at DESC LIMIT 1
    """, (symbol, market, pred_date)).fetchone()
    conn.close()
    
    if row is None:
        return None
    
    pred_records = json.loads(row[0])
    pred_df = pd.DataFrame(pred_records)
    pred_df.index = pd.to_datetime(pred_df["date"])
    pred_df.drop(columns=["date"], inplace=True)
    
    return {
        "pred_df": pred_df,
        "last_hist_date": row[1],
        "last_hist_close": float(row[2]),
        "temperature": float(row[3]),
        "pred_len": int(row[4]),
        "created_at": row[5],
    }


def get_signal_symbols(market: str = "US") -> list:
    """从今日扫描信号中提取候选股票代码"""
    db_path = os.path.join(V3_DIR, "db", "coral_creek.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    today = datetime.now().strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT DISTINCT symbol FROM signals 
        WHERE scan_date = ? AND market = ?
        ORDER BY symbol
    """, (today, market)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def predict_single(engine, cache, symbol: str, market: str = "US", pred_len: int = 20, temperature: float = 0.5):
    """对单只股票运行 Kronos 预测并缓存"""
    import pandas as pd
    
    print(f"  📊 获取 {symbol} 历史数据...")
    df = cache.get_stock_history(symbol, market=market)
    
    if df is None or len(df) < 60:
        print(f"  ⚠️ {symbol} 数据不足, 跳过 (len={len(df) if df is not None else 0})")
        return False
    
    # 准备输入
    df_input = df.copy()
    df_input.columns = [c.lower() for c in df_input.columns]
    if "date" in df_input.columns:
        df_input.rename(columns={"date": "timestamps"}, inplace=True)
    df_input = df_input.tail(400)
    
    print(f"  🧠 Kronos 推理中 ({len(df_input)} 根K线 → {pred_len}天预测)...")
    t0 = time.time()
    pred_df = engine.predict_future_klines(df_input, pred_len=pred_len, temperature=temperature, top_p=0.8)
    elapsed = time.time() - t0
    
    if pred_df is None:
        print(f"  ❌ {symbol} 预测失败")
        return False
    
    # 保存
    last_close = float(df_input["close"].iloc[-1])
    last_date = str(df_input["timestamps"].iloc[-1])[:10] if "timestamps" in df_input.columns else "unknown"
    save_prediction(symbol, market, pred_df, last_date, last_close, pred_len, temperature)
    
    pred_chg = (float(pred_df["Close"].iloc[-1]) / last_close - 1) * 100
    direction = "📈" if pred_chg > 0 else "📉"
    print(f"  ✅ {symbol} 完成 ({elapsed:.1f}s) {direction} 预测{pred_len}日变幅: {pred_chg:+.2f}%")
    return True


def main():
    parser = argparse.ArgumentParser(description="Kronos 批量预计算")
    parser.add_argument("symbols", nargs="*", help="要预测的股票代码列表")
    parser.add_argument("--from-signals", action="store_true", help="从今日扫描信号中自动提取")
    parser.add_argument("--market", default="US", help="市场 (US/CN)")
    parser.add_argument("--pred-len", type=int, default=20, help="预测天数")
    parser.add_argument("--temperature", type=float, default=0.5, help="随机度")
    args = parser.parse_args()
    
    symbols = list(args.symbols)
    
    if args.from_signals:
        sig_symbols = get_signal_symbols(args.market)
        print(f"📡 从今日扫描信号中发现 {len(sig_symbols)} 只股票")
        symbols = list(set(symbols + sig_symbols))
    
    if not symbols:
        symbols = ["HIMS", "AAPL", "NVDA", "TSLA", "PLTR"]  # 默认热门
        print(f"ℹ️ 未指定股票, 使用默认列表: {symbols}")
    
    print(f"\n🪐 Kronos 批量预计算启动")
    print(f"   股票数量: {len(symbols)}")
    print(f"   预测天数: {args.pred_len}")
    print(f"   温度参数: {args.temperature}")
    print(f"   缓存路径: {CACHE_DB}")
    print()
    
    # 初始化
    init_cache_db()
    cache = DataCache()
    
    print("🚀 加载 Kronos 大模型引擎...")
    t_start = time.time()
    engine = get_kronos_engine()
    print(f"✅ 引擎加载完成 ({time.time() - t_start:.1f}s)\n")
    
    success = 0
    fail = 0
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")
        try:
            if predict_single(engine, cache, sym, args.market, args.pred_len, args.temperature):
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  ❌ {sym} 异常: {e}")
            fail += 1
    
    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"🏁 批量预计算完成!")
    print(f"   成功: {success} | 失败: {fail} | 总耗时: {total_time:.1f}s")
    print(f"   结果已保存到: {CACHE_DB}")


if __name__ == "__main__":
    main()
