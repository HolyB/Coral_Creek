"""
Coral Creek 入口
默认运行 V2 (稳定版)

Streamlit Cloud 部署:
- 设置 Main file path 为: versions/v2/app.py (推荐)
- 或者使用此文件作为入口

本地开发:
- V2: cd versions/v2 && streamlit run app.py --server.port 8503
- V3: cd versions/v3 && streamlit run app.py --server.port 8504
"""

import sys
import os

# 默认运行 V2
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_PATH = os.path.join(ROOT_DIR, "versions", "v2")

# 添加 V2 路径
if V2_PATH not in sys.path:
    sys.path.insert(0, V2_PATH)

# 切换工作目录到 V2
os.chdir(V2_PATH)

# 运行 V2 app
import runpy
runpy.run_path(os.path.join(V2_PATH, "app.py"), run_name="__main__")
