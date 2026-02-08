"""Root Streamlit pages wrapper for V3 qlib mining page."""

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V3_PAGE = ROOT / "versions" / "v3" / "pages" / "qlib_mining.py"

# Ensure V3 is on sys.path and run the target page in-place
os.chdir(ROOT / "versions" / "v3")
runpy.run_path(str(V3_PAGE), run_name="__main__")
