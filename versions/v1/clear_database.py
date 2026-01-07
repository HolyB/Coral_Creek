#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""清空数据库"""
import sys
import os

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_manager import StockDatabase
import sqlite3

print("=" * 60)
print("清空数据库")
print("=" * 60)

db = StockDatabase()
cursor = db.conn.cursor()

# 获取当前记录数
cursor.execute('SELECT COUNT(*) FROM scan_results')
count_before = cursor.fetchone()[0]
print(f"\n当前记录数: {count_before:,}")

# 确认
confirm = input("\n确认要清空所有数据吗？(yes/no): ")
if confirm.lower() != 'yes':
    print("操作已取消")
    db.close()
    sys.exit(0)

# 清空表
print("\n正在清空数据...")
cursor.execute('DELETE FROM scan_results')
db.conn.commit()

# 验证
cursor.execute('SELECT COUNT(*) FROM scan_results')
count_after = cursor.fetchone()[0]
print(f"清空后记录数: {count_after:,}")

if count_after == 0:
    print("\n✅ 数据库已清空")
else:
    print("\n⚠️ 警告：清空后仍有数据，请检查")

db.close()
print("\n" + "=" * 60)


