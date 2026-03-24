#!/bin/bash
# 每周 ML 训练 + 推送模型
# crontab: 0 10 * * 0 (每周日 10 AM PT)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V3_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$(dirname "$V3_DIR")")"
LOG_DIR="$V3_DIR/logs"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/ml_train_${DATE}.log"

# 加载环境变量
if [ -f "$V3_DIR/.env" ]; then
    export $(grep -v '^#' "$V3_DIR/.env" | xargs)
fi

echo "========================================" >> "$LOG_FILE"
echo "🧠 Weekly ML Training - $DATE" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$V3_DIR"

# 1. XGBoost 训练 (standard + penny)
echo "🚀 Training XGBoost models..." >> "$LOG_FILE"
/Users/bertwang/miniconda3/bin/python3 -u ml/pipeline.py \
    --market US --days 9999 --all-tiers >> "$LOG_FILE" 2>&1

# 2. MMoE 训练
echo "🧠 Training MMoE models..." >> "$LOG_FILE"
/Users/bertwang/miniconda3/bin/python3 -u -c "
import warnings; warnings.filterwarnings('ignore')
from ml.pipeline import MLPipeline
from ml.models.mmoe import MMoEPredictor, ALL_TASK_DEFS

BEST_TASKS = [t for t in ALL_TASK_DEFS if t['name'] != 'volatility']
kwargs = dict(
    num_experts=4, expert_hidden=64, expert_out=32, tower_hidden=16,
    dropout=0.2, lr=5e-4, weight_decay=1e-3,
    epochs=200, batch_size=128, patience=25,
    task_defs=BEST_TASKS,
)

for tier in ['standard', 'penny']:
    print(f'\\n=== MMoE {tier} ===')
    p = MLPipeline(market='US', days_back=9999, price_tier=tier)
    X, ret, dd, grp, fn, info = p.prepare_dataset()
    mmoe = MMoEPredictor(**kwargs)
    r = mmoe.train(X, ret, fn, grp, dd)
    suffix = '_penny' if tier == 'penny' else ''
    mmoe.save(f'ml/saved_models/v2_us{suffix}_mmoe')
    print(f'  dir={r[\"dir_accuracy\"]:.1%}, mae5={r[\"mae_5d\"]:.2f}%')
" >> "$LOG_FILE" 2>&1

# 3. 推送模型到 GitHub
echo "📤 Pushing models to GitHub..." >> "$LOG_FILE"
cd "$REPO_DIR"

# 安全检查：不要动 db 文件
git add versions/v3/ml/saved_models/ >> "$LOG_FILE" 2>&1

if git diff --staged --quiet; then
    echo "No model changes to push" >> "$LOG_FILE"
else
    git commit -m "🧠 Local retrain: ML models $DATE" >> "$LOG_FILE" 2>&1
    git push origin main >> "$LOG_FILE" 2>&1 || {
        echo "⚠️ Push failed, will retry next week" >> "$LOG_FILE"
    }
fi

echo "" >> "$LOG_FILE"
echo "✅ ML training completed: $(date)" >> "$LOG_FILE"

# 清理 30 天前的日志
find "$LOG_DIR" -name "ml_train_*.log" -mtime +30 -delete 2>/dev/null
