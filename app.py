"""
Coral Creek 入口
现在运行 V3 (最新版)

Streamlit Cloud 部署:
- 使用此文件作为入口，自动运行 V3

本地开发:
- V3: cd versions/v3 && streamlit run app.py --server.port 8504
"""

import sys
import os
import runpy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_PATH = os.path.join(ROOT_DIR, "versions", "v3")

# 添加 V3 路径
if V3_PATH not in sys.path:
    sys.path.insert(0, V3_PATH)

script_path = os.path.join(V3_PATH, "app.py")
runpy.run_path(script_path, run_name="__main__")
