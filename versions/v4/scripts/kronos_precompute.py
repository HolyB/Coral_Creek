"""
Kronos 预计算脚本
==================
独立于 Streamlit 运行，提前为指定股票批量计算 Kronos 预测结果，
将结果缓存到 SQLite 数据库，网页端直接读取即可秒开。

用法:
    # 预测指定股票
    python scripts/kronos_precompute.py HIMS AAPL NVDA TSLA

    # 预测今日扫描出的所有股票 (含 BLUE 信号)
    python scripts/kronos_precompute.py --from-scan

    # 仅预测有 BLUE 信号的
    python scripts/kronos_precompute.py --from-scan --blue-only

    # 指定市场
    python scripts/kronos_precompute.py --from-scan --market CN

    # 跑全量 (美股+A股)
    python scripts/kronos_precompute.py --from-scan --market US --from-scan --market CN

    # 性能基准测试 (只跑前 N 只)
    python scripts/kronos_precompute.py --from-scan --benchmark 50
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


def get_scan_symbols(market: str = "US", blue_only: bool = False) -> list:
    """从最近一次扫描结果中提取股票代码"""
    db_path = os.path.join(V3_DIR, "db", "coral_creek.db")
    if not os.path.exists(db_path):
        print(f"  ⚠️ 数据库不存在: {db_path}")
        return []
    conn = sqlite3.connect(db_path)
    
    # 查找最近的扫描日期
    latest = conn.execute("""
        SELECT MAX(scan_date) FROM scan_results WHERE market=?
    """, (market,)).fetchone()[0]
    
    if latest is None:
        conn.close()
        return []
    
    if blue_only:
        rows = conn.execute("""
            SELECT DISTINCT symbol FROM scan_results 
            WHERE scan_date = ? AND market = ? AND blue_daily > 0
            ORDER BY symbol
        """, (latest, market)).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT symbol FROM scan_results 
            WHERE scan_date = ? AND market = ?
            ORDER BY symbol
        """, (latest, market)).fetchall()
    conn.close()
    print(f"  📡 [{market}] 扫描日期 {latest}, 找到 {len(rows)} 只股票" + (" (仅BLUE)" if blue_only else ""))
    return [r[0] for r in rows]


def predict_single(engine, cache, symbol: str, market: str = "US", pred_len: int = 20, temperature: float = 0.5):
    """对单只股票运行 Kronos 预测并缓存"""
    import pandas as pd
    
    df = cache.get_stock_history(symbol, market=market)
    
    if df is None or len(df) < 60:
        return False, 0.0
    
    # 准备输入
    df_input = df.copy()
    df_input.columns = [c.lower() for c in df_input.columns]
    if "date" in df_input.columns:
        df_input.rename(columns={"date": "timestamps"}, inplace=True)
    df_input = df_input.tail(400)
    
    t0 = time.time()
    pred_df = engine.predict_future_klines(df_input, pred_len=pred_len, temperature=temperature, top_p=0.8)
    elapsed = time.time() - t0
    
    if pred_df is None:
        return False, elapsed
    
    # 保存
    last_close = float(df_input["close"].iloc[-1])
    last_date = str(df_input["timestamps"].iloc[-1])[:10] if "timestamps" in df_input.columns else "unknown"
    save_prediction(symbol, market, pred_df, last_date, last_close, pred_len, temperature)
    
    pred_chg = (float(pred_df["Close"].iloc[-1]) / last_close - 1) * 100
    direction = "📈" if pred_chg > 0 else "📉"
    print(f"  ✅ {symbol} ({elapsed:.1f}s) {direction} {pred_chg:+.2f}%")
    return True, elapsed


def main():
    parser = argparse.ArgumentParser(description="Kronos 批量预计算")
    parser.add_argument("symbols", nargs="*", help="要预测的股票代码列表")
    parser.add_argument("--from-scan", action="store_true", help="从最近一次扫描结果中提取")
    parser.add_argument("--blue-only", action="store_true", help="仅预测有 BLUE 信号的股票")
    parser.add_argument("--market", default="US", help="市场 (US/CN)")
    parser.add_argument("--pred-len", type=int, default=20, help="预测天数")
    parser.add_argument("--temperature", type=float, default=0.5, help="随机度")
    parser.add_argument("--benchmark", type=int, default=0, help="基准测试模式: 只跑前 N 只")
    args = parser.parse_args()
    
    symbols = list(args.symbols)
    
    if args.from_scan:
        scan_symbols = get_scan_symbols(args.market, args.blue_only)
        symbols = list(set(symbols + scan_symbols))
    
    if not symbols:
        symbols = ["HIMS", "AAPL", "NVDA", "TSLA", "PLTR"]
        print(f"ℹ️ 未指定股票, 使用默认列表: {symbols}")
    
    if args.benchmark > 0:
        symbols = symbols[:args.benchmark]
        print(f"⚡ 基准测试模式: 只处理前 {args.benchmark} 只")
    
    print(f"\n🪐 Kronos 批量预计算启动")
    print(f"   市场: {args.market}")
    print(f"   股票数量: {len(symbols)}")
    print(f"   预测天数: {args.pred_len}")
    print(f"   温度参数: {args.temperature}")
    print(f"   缓存路径: {CACHE_DB}")
    print()
    
    # 初始化
    init_cache_db()
    cache = DataCache()
    
    print("🚀 加载 Kronos 大模型引擎...")
    t_engine = time.time()
    engine = get_kronos_engine()
    engine_time = time.time() - t_engine
    print(f"✅ 引擎加载完成 ({engine_time:.1f}s)\n")
    
    t_start = time.time()
    success = 0
    fail = 0
    skip = 0
    times = []
    
    for i, sym in enumerate(symbols, 1):
        try:
            ok, elapsed = predict_single(engine, cache, sym, args.market, args.pred_len, args.temperature)
            if ok:
                success += 1
                times.append(elapsed)
            else:
                skip += 1
        except Exception as e:
            print(f"  ❌ {sym}: {e}")
            fail += 1
        
        # 每 50 只输出一次进度
        if i % 50 == 0:
            elapsed_total = time.time() - t_start
            avg = sum(times) / len(times) if times else 0
            eta = avg * (len(symbols) - i)
            print(f"\n--- 进度: {i}/{len(symbols)} | 成功: {success} | 跳过: {skip} | 失败: {fail} | "
                  f"已用: {elapsed_total:.0f}s | 平均: {avg:.2f}s/只 | 预计剩余: {eta:.0f}s ---\n")
    
    total_time = time.time() - t_start
    avg_time = sum(times) / len(times) if times else 0
    
    print(f"\n{'='*60}")
    print(f"🏁 Kronos 批量预计算完成!")
    print(f"   市场: {args.market}")
    print(f"   成功: {success} | 跳过(数据不足): {skip} | 失败: {fail}")
    print(f"   推理总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   引擎加载耗时: {engine_time:.1f}s")
    print(f"   平均每只: {avg_time:.2f}s")
    if success > 0:
        print(f"   全量预估 (500只): {500 * avg_time:.0f}s ({500 * avg_time / 60:.1f}min)")
        print(f"   全量预估 (1000只): {1000 * avg_time:.0f}s ({1000 * avg_time / 60:.1f}min)")
    print(f"   结果已保存到: {CACHE_DB}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
