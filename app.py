"""
Coral Creek 入口
现在运行 V3 (最新版)

Streamlit Cloud 部署:
- 使用此文件作为入口，自动运行 V3

本地开发:
- V3: cd versions/v3 && streamlit run app.py --server.port 8504
- V2: cd versions/v2 && streamlit run app.py --server.port 8503 (备用)
"""

import sys
import os
import traceback

# 运行 V3
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_PATH = os.path.join(ROOT_DIR, "versions", "v3")

# 添加 V3 路径
if V3_PATH not in sys.path:
    sys.path.insert(0, V3_PATH)

# 切换工作目录到 V3
os.chdir(V3_PATH)

# 运行 V3 app（失败时在页面显示可读错误，避免 Streamlit Cloud 只给 Server Error）
import runpy
try:
    runpy.run_path(os.path.join(V3_PATH, "app.py"), run_name="__main__")
except BaseException as e:
    original_tb = traceback.format_exc()
    try:
        import streamlit as st
        st.error(f"应用启动失败: {e}")
        st.code(original_tb)
        st.info("请把上面的错误栈发给我，我会立刻修复。")
    except Exception:
        # 最后兜底：保证日志里有完整栈
        print("Coral Creek bootstrap failed:")
        print(original_tb)
        print("Secondary error while rendering Streamlit fallback:")
        print(traceback.format_exc())
        raise
