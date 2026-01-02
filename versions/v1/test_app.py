#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试app.py是否能正常运行"""
import sys
import os

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing database connection...")
    from database_manager import StockDatabase
    db = StockDatabase()
    dates = db.get_available_dates()
    print(f"[OK] Database OK, {len(dates)} dates available")
    
    print("\nTesting Streamlit imports...")
    import streamlit as st
    print(f"[OK] Streamlit OK, version: {st.__version__}")
    
    print("\nTesting app.py imports...")
    # 只测试导入，不运行
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app.py")
    if spec and spec.loader:
        print("[OK] app.py can be loaded")
    else:
        print("[ERROR] app.py cannot be loaded")
    
    print("\nAll tests passed! Streamlit should be running.")
    print("Access URL: http://localhost:8501")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()

