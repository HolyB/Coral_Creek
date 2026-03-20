#!/bin/bash
# ==============================================================
# XGB+MMoE 每日定时任务 v2
# ==============================================================
# 调度规则 (Pacific Time):
#
# US (周一-周五):
#   10:00 AM  数据拉取 + 增量特征 (~40min)
#   12:30 PM  预测 + 推送 (盘前30分, ~90s)
#   1:30 PM   盘后再跑一次 (更新收益 + 推送)
#
# CN (周日-周四):
#   8:00 PM   数据拉取 + 增量特征 (~40min)
#   10:30 PM  预测 + 推送 (盘前30分, ~90s)
#   11:30 PM  盘后再跑一次
# ==============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V3_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/Users/bertwang/miniconda3/bin/python3"
LOG_DIR="/tmp/ml_pipeline_logs"
mkdir -p "$LOG_DIR"

# ===== 数据 + 特征更新 (耗时~40min, 提前跑) =====
update_features() {
    local MARKET="$1"
    local TS=$(date +"%Y%m%d_%H%M")
    local LOG="$LOG_DIR/${MARKET}_features_${TS}.log"

    echo "======================================" | tee -a "$LOG"
    echo "📦 ${MARKET} Feature Update @ $(date)" | tee -a "$LOG"
    echo "======================================" | tee -a "$LOG"

    cd "$V3_DIR"

    # Step 1: 拉最新行情
    echo "📥 Step 1: Update market data..." | tee -a "$LOG"
    if [ "$MARKET" = "US" ]; then
        # US: Polygon grouped daily API
        PYTHONPATH="$V3_DIR" $PYTHON -u -c "
import os, sqlite3, requests
from dotenv import load_dotenv
load_dotenv('$V3_DIR/.env')
API_KEY = os.getenv('POLYGON_API_KEY')
from datetime import datetime, timedelta
DB = '$V3_DIR/db/stock_history.db'
conn = sqlite3.connect(DB)
# Check latest date
mx = conn.execute(\"SELECT MAX(trade_date) FROM stock_history WHERE market='US'\").fetchone()[0]
print(f'US latest: {mx}')
# Pull next business days
from datetime import date
today = date.today()
for delta in range(1, 5):
    d = (today - timedelta(days=delta)).strftime('%Y-%m-%d')
    if d <= mx: continue
    dapi = d.replace('-','')
    url = f'https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d}?adjusted=true&apiKey={API_KEY}'
    r = requests.get(url, timeout=30)
    data = r.json()
    if data.get('resultsCount', 0) > 0:
        for bar in data['results']:
            conn.execute('INSERT OR IGNORE INTO stock_history (symbol,market,trade_date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)',
                (bar['T'], 'US', d, bar['o'], bar['h'], bar['l'], bar['c'], bar['v']))
        conn.commit()
        print(f'US {d}: +{data[\"resultsCount\"]} rows')
conn.close()
" >> "$LOG" 2>&1 || echo "⚠️ US data update failed, continuing..." | tee -a "$LOG"
    else
        # CN: Tushare daily API
        PYTHONPATH="$V3_DIR" $PYTHON -u -c "
import os, sqlite3, tushare as ts
from dotenv import load_dotenv
load_dotenv('$V3_DIR/.env')
ts.set_token(os.getenv('TUSHARE_TOKEN'))
pro = ts.pro_api()
DB = '$V3_DIR/db/stock_history.db'
conn = sqlite3.connect(DB)
mx = conn.execute(\"SELECT MAX(trade_date) FROM stock_history WHERE market='CN'\").fetchone()[0]
print(f'CN latest: {mx}')
from datetime import datetime, timedelta, date
today = date.today()
for delta in range(0, 5):
    d = (today - timedelta(days=delta))
    ds = d.strftime('%Y%m%d')
    ds_db = d.strftime('%Y-%m-%d')
    if ds_db <= mx: continue
    try:
        df = pro.daily(trade_date=ds)
        if len(df) > 0:
            for _, r in df.iterrows():
                td = f'{r[\"trade_date\"][:4]}-{r[\"trade_date\"][4:6]}-{r[\"trade_date\"][6:]}'
                conn.execute('INSERT OR IGNORE INTO stock_history (symbol,market,trade_date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)',
                    (r['ts_code'], 'CN', td, r['open'], r['high'], r['low'], r['close'], r['vol']))
            conn.commit()
            print(f'CN {ds_db}: +{len(df)} rows')
    except Exception as e:
        print(f'CN {ds_db}: {e}')
conn.close()
" >> "$LOG" 2>&1 || echo "⚠️ CN data update failed, continuing..." | tee -a "$LOG"
    fi

    # Step 2: 增量更新 npz 特征
    echo "📊 Step 2: Incremental feature update..." | tee -a "$LOG"
    $PYTHON /tmp/gen_features_incremental.py "$MARKET" >> "$LOG" 2>&1 || echo "⚠️ Feature update failed" | tee -a "$LOG"

    echo "✅ Features updated @ $(date)" | tee -a "$LOG"
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null
}

# ===== 预测 + 推送 (快速, ~90s) =====
run_predict() {
    local MARKET="$1"
    local MODE="$2"
    local TS=$(date +"%Y%m%d_%H%M")
    local LOG="$LOG_DIR/${MARKET}_${MODE}_${TS}.log"

    echo "======================================" | tee -a "$LOG"
    echo "🤖 ${MARKET} ${MODE} Predict @ $(date)" | tee -a "$LOG"
    echo "======================================" | tee -a "$LOG"

    cd "$V3_DIR"
    EXTRA_FLAGS=""
    if [ "$MODE" = "trade" ]; then
        EXTRA_FLAGS="--trade"
    fi
    PYTHONPATH="$V3_DIR" $PYTHON scripts/ml_daily_pipeline.py --market "$MARKET" $EXTRA_FLAGS >> "$LOG" 2>&1

    echo "✅ Done @ $(date)" | tee -a "$LOG"
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null
}

# ===== 命令行入口 =====
case "${1:-help}" in
    us_features) update_features US ;;
    cn_features) update_features CN ;;
    us_pre)      run_predict US pre_close ;;
    us_trade)    run_predict US trade ;;
    us_post)     run_predict US post_close ;;
    cn_pre)      run_predict CN pre_close ;;
    cn_trade)    run_predict CN trade ;;
    cn_post)     run_predict CN post_close ;;
    both)
        update_features US
        run_predict US post_close
        update_features CN
        run_predict CN post_close
        ;;
    install)
        echo "📅 Installing crontab..."
        crontab -l 2>/dev/null | grep -v "ml_cron_job.sh" > /tmp/crontab_tmp || true

        cat >> /tmp/crontab_tmp << EOF
# === XGB+MMoE Daily Pipeline v2 ===
# === XGB+MMoE Daily Pipeline v2 ===
# --- US (Mon-Fri) ---
# 4:00 AM PT (7:00 AM ET): 拉数据 + 增量特征 (~40min)
0 4 * * 1-5 $SCRIPT_DIR/ml_cron_job.sh us_features >> $LOG_DIR/cron.log 2>&1
# 6:00 AM PT (9:00 AM ET): 盘前预测+推送 (~90s)
0 6 * * 1-5 $SCRIPT_DIR/ml_cron_job.sh us_pre >> $LOG_DIR/cron.log 2>&1
# 6:31 AM PT (9:31 AM ET): 开盘后自动交易 (Alpaca)
31 6 * * 1-5 $SCRIPT_DIR/ml_cron_job.sh us_trade >> $LOG_DIR/cron.log 2>&1
# 12:30 PM PT (3:30 PM ET): 收盘前30分 预测+推送
30 12 * * 1-5 $SCRIPT_DIR/ml_cron_job.sh us_pre >> $LOG_DIR/cron.log 2>&1
# 1:30 PM PT (4:30 PM ET): 盘后 更新收益+推送
30 13 * * 1-5 $SCRIPT_DIR/ml_cron_job.sh us_post >> $LOG_DIR/cron.log 2>&1
# --- CN (Sun-Thu) ---
# 8:00 PM PT (11:00 AM CST+1): 拉数据 + 增量特征
0 20 * * 0-4 $SCRIPT_DIR/ml_cron_job.sh cn_features >> $LOG_DIR/cron.log 2>&1
# 10:30 PM PT (2:30 PM CST+1): 盘前30分 预测+推送
30 22 * * 0-4 $SCRIPT_DIR/ml_cron_job.sh cn_pre >> $LOG_DIR/cron.log 2>&1
# 11:01 PM PT (3:01 PM CST+1): 收盘后虚拟盘交易
1 23 * * 0-4 $SCRIPT_DIR/ml_cron_job.sh cn_trade >> $LOG_DIR/cron.log 2>&1
# 11:30 PM PT (3:30 PM CST+1): 盘后 更新收益+推送
30 23 * * 0-4 $SCRIPT_DIR/ml_cron_job.sh cn_post >> $LOG_DIR/cron.log 2>&1
EOF
        crontab /tmp/crontab_tmp
        rm /tmp/crontab_tmp
        echo "✅ Crontab installed:"
        crontab -l | grep "ml_cron_job"
        ;;
    status)
        echo "📊 Recent logs:"
        ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -10
        echo ""
        echo "📅 Crontab entries:"
        crontab -l 2>/dev/null | grep "ml_cron_job" || echo "  (none)"
        echo ""
        echo "📦 NPZ status:"
        for m in us cn; do
            f="/tmp/${m}_daily_full.npz"
            if [ -f "$f" ]; then
                echo "  $m: $(ls -lh "$f" | awk '{print $5}') modified $(stat -f '%Sm' -t '%m-%d %H:%M' "$f")"
            fi
        done
        ;;
    help|*)
        echo "Usage: $0 {command}"
        echo ""
        echo "Commands:"
        echo "  us_features - US 数据拉取 + 增量特征更新 (~40min)"
        echo "  cn_features - CN 数据拉取 + 增量特征更新"
        echo "  us_pre      - US 盘前30分 预测+推送 (~90s)"
        echo "  us_post     - US 盘后 预测+推送"
        echo "  cn_pre      - CN 盘前30分"
        echo "  cn_post     - CN 盘后"
        echo "  both        - US+CN 全流程"
        echo "  install     - 安装到 crontab"
        echo "  status      - 查看状态"
        ;;
esac
