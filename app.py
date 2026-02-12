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
import time
import importlib

# 运行 V3
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_PATH = os.path.join(ROOT_DIR, "versions", "v3")

# 添加 V3 路径
if V3_PATH not in sys.path:
    sys.path.insert(0, V3_PATH)

def _clear_project_modules(trigger_module: str = ""):
    """清理 V3 相关模块缓存，规避 py3.13 下偶发导入竞态 KeyError。"""
    trigger = (trigger_module or "").strip()
    if trigger.startswith("'") and trigger.endswith("'"):
        trigger = trigger[1:-1]

    prefixes = (
        "components",
        "db",
        "services",
        "strategies",
        "ml",
        "scripts",
    )
    singles = {
        "chart_utils",
        "indicator_utils",
        "backtester",
        "data_fetcher",
        "advanced_charts",
    }

    for name, mod in list(sys.modules.items()):
        if not name or name == "__main__":
            continue

        drop = False
        if trigger and (name == trigger or name.startswith(f"{trigger}.")):
            drop = True
        elif name in singles:
            drop = True
        elif any(name == p or name.startswith(f"{p}.") for p in prefixes):
            drop = True
        else:
            mod_file = getattr(mod, "__file__", "") or ""
            if mod_file:
                try:
                    if os.path.abspath(mod_file).startswith(os.path.abspath(V3_PATH)):
                        drop = True
                except Exception:
                    pass

        if drop:
            sys.modules.pop(name, None)

    importlib.invalidate_caches()


# 运行 V3 app（失败时在页面显示可读错误，绝不二次 raise）
import runpy
script_path = os.path.join(V3_PATH, "app.py")
max_attempts = 12
last_exc = None
last_tb = ""

for attempt in range(1, max_attempts + 1):
    try:
        runpy.run_path(script_path, run_name="__main__")
        last_exc = None
        break
    except Exception as e:
        last_exc = e
        last_tb = traceback.format_exc()
        print(f"Coral Creek bootstrap failed (attempt {attempt}/{max_attempts}): {e}")
        print(last_tb)

        recoverable = isinstance(e, KeyError) or isinstance(e, ImportError)
        if (not recoverable) or attempt >= max_attempts:
            break

        _clear_project_modules(str(e))
        time.sleep(0.05)

if last_exc is not None:
    try:
        import streamlit as st
        st.error(f"应用启动失败: {last_exc}")
        st.code(last_tb)
        st.info("启动器已执行自动重试与模块清理。请把这段错误栈发我继续修复。")
    except Exception as render_err:
        # 只打印，不再抛出，避免 Streamlit Cloud healthz 失败
        print("Fallback rendering failed:", render_err)
