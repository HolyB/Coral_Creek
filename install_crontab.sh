#!/bin/bash
# 增强版股票扫描系统 - Crontab安装脚本
# 使用方法: bash install_crontab.sh

echo "======================================"
echo "   股票扫描系统 Crontab 安装脚本"
echo "======================================"

# 检查crontab是否可用
if ! command -v crontab &> /dev/null; then
    echo "❌ 错误: crontab 命令不可用"
    echo "请确保您在支持cron的环境中运行此脚本 (WSL/Linux/MacOS)"
    exit 1
fi

# 获取当前工作目录
CURRENT_DIR=$(pwd)
echo "📁 当前工作目录: $CURRENT_DIR"

# 检查必要文件是否存在
if [ ! -f "$CURRENT_DIR/scan_cn_signals_multi_thread_tushare.py" ]; then
    echo "❌ 错误: A股扫描脚本不存在"
    exit 1
fi

if [ ! -f "$CURRENT_DIR/scan_hk_signals_multi_thread_tushare.py" ]; then
    echo "❌ 错误: 港股扫描脚本不存在"
    exit 1
fi

echo "✅ 扫描脚本检查通过"

# 备份现有的crontab
echo "📋 备份现有crontab..."
crontab -l > crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || echo "没有现有的crontab任务"

# 创建新的crontab内容
cat > temp_crontab.txt << EOF
# 增强版股票扫描定时任务 (基于北京时间)
# 生成时间: $(date)

# 环境设置
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
TZ=Asia/Shanghai

# 盘前早期扫描 - 8:30 (周一到周五)
30 8 * * 1-5 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 300 --max_workers 5 --timing "盘前早期"

# 盘前扫描 - 9:00 (周一到周五)  
0 9 * * 1-5 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 400 --max_workers 8 --timing "盘前"

# 上午盘中扫描 - 10:30 (周一到周五)
30 10 * * 1-5 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 500 --max_workers 10 --timing "上午盘中"

# 下午盘中扫描 - 14:00 (周一到周五)
0 14 * * 1-5 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 500 --max_workers 10 --timing "下午盘中"

# 盘后扫描 - 15:30 (周一到周五)
30 15 * * 1-5 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 400 --max_workers 8 --timing "盘后"

# 盘后深度扫描 - 16:30 (周一到周五)
30 16 * * 1-5 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 400 --max_workers 8 --timing "盘后深度"

# 港股盘前扫描 - 9:15 (周一到周五)
15 9 * * 1-5 cd $CURRENT_DIR && python scan_hk_signals_multi_thread_tushare.py --batch_size 250 --max_workers 8 --timing "港股盘前"

# 港股盘中扫描 - 11:00 (周一到周五)
0 11 * * 1-5 cd $CURRENT_DIR && python scan_hk_signals_multi_thread_tushare.py --batch_size 300 --max_workers 10 --timing "港股盘中"

# 港股下午扫描 - 14:30 (周一到周五)
30 14 * * 1-5 cd $CURRENT_DIR && python scan_hk_signals_multi_thread_tushare.py --batch_size 300 --max_workers 10 --timing "港股下午"

# 港股盘后扫描 - 16:30 (周一到周五)
30 16 * * 1-5 cd $CURRENT_DIR && python scan_hk_signals_multi_thread_tushare.py --batch_size 250 --max_workers 8 --timing "港股盘后"

# 周末综合扫描 - 周六 10:00
0 10 * * 6 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 600 --max_workers 12 --timing "周六A股"

# 周末港股扫描 - 周六 10:30
30 10 * * 6 cd $CURRENT_DIR && python scan_hk_signals_multi_thread_tushare.py --batch_size 400 --max_workers 12 --timing "周六港股"

# 周日准备扫描 - 20:00
0 20 * * 0 cd $CURRENT_DIR && python scan_cn_signals_multi_thread_tushare.py --batch_size 600 --max_workers 12 --timing "周日准备"

EOF

# 安装新的crontab
echo "⚙️  安装新的crontab任务..."
crontab temp_crontab.txt

if [ $? -eq 0 ]; then
    echo "✅ Crontab任务安装成功！"
    echo ""
    echo "📅 定时任务安排 (北京时间):"
    echo "  工作日 A股:"
    echo "    08:30 - 盘前早期扫描"
    echo "    09:00 - 盘前扫描"
    echo "    10:30 - 上午盘中扫描"
    echo "    14:00 - 下午盘中扫描"
    echo "    15:30 - 盘后扫描"
    echo "    16:30 - 盘后深度扫描"
    echo ""
    echo "  工作日 港股:"
    echo "    09:15 - 盘前扫描"
    echo "    11:00 - 盘中扫描"
    echo "    14:30 - 下午扫描"
    echo "    16:30 - 盘后扫描"
    echo ""
    echo "  周末:"
    echo "    周六 10:00 - A股综合扫描"
    echo "    周六 10:30 - 港股扫描"
    echo "    周日 20:00 - 准备扫描"
    echo ""
    echo "🔍 查看当前任务: crontab -l"
    echo "📝 查看cron日志: sudo tail -f /var/log/cron"
    echo "🗂️  备份文件已保存"
else
    echo "❌ Crontab任务安装失败！"
    exit 1
fi

# 清理临时文件
rm -f temp_crontab.txt

echo ""
echo "======================================"
echo "  安装完成！系统将自动执行定时任务"
echo "======================================" 