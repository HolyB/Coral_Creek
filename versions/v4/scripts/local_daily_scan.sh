#!/bin/bash
# 每日本地扫描 - 自动追加当天数据到本地 db
# crontab: 30 17 * * 1-5 (每个工作日 5:30 PM PT，美股收盘后)
#          00 20 * * 1-5 (每个工作日 8:00 PM PT，A股数据就绪后)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V3_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$V3_DIR/logs"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/daily_scan_${DATE}.log"

# 加载环境变量
if [ -f "$V3_DIR/.env" ]; then
    export $(grep -v '^#' "$V3_DIR/.env" | xargs)
fi

echo "========================================" >> "$LOG_FILE"
echo "🦅 Local Daily Scan - $DATE" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$V3_DIR"

# 0. 先从 Supabase 同步缺失数据（自动补缺）
echo "📥 Syncing from Supabase..." >> "$LOG_FILE"
PYTHONPATH="$V3_DIR" /Users/bertwang/miniconda3/bin/python3 -u scripts/sync_from_supabase.py \
    --days 30 >> "$LOG_FILE" 2>&1 || true

# 1. US 扫描
echo "📊 Scanning US market..." >> "$LOG_FILE"
/Users/bertwang/miniconda3/bin/python3 -u services/scan_service.py \
    --date "$DATE" --market US --workers 20 >> "$LOG_FILE" 2>&1 || true

# 2. CN 扫描
echo "📊 Scanning CN market..." >> "$LOG_FILE"
/Users/bertwang/miniconda3/bin/python3 -u services/scan_service.py \
    --date "$DATE" --market CN --workers 12 >> "$LOG_FILE" 2>&1 || true

echo "" >> "$LOG_FILE"
echo "✅ Daily scan completed: $(date)" >> "$LOG_FILE"

# 清理 7 天前的日志
find "$LOG_DIR" -name "daily_scan_*.log" -mtime +7 -delete 2>/dev/null
