"""Qlib 因子/策略挖掘页面。"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.qlib_integration import check_qlib_status

st.set_page_config(page_title="Qlib 挖掘", page_icon="🧠", layout="wide")
st.title("🧠 Qlib 因子与策略挖掘")
st.caption("一键查看 Alpha 因子排名、策略网格结果，以及大/中/小市值分层效果")


def _model_dir(market: str) -> Path:
    return project_root / "ml" / "saved_models" / f"qlib_{market.lower()}"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run_command(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=1800,
            env=os.environ.copy(),
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, out.strip()
    except Exception as exc:
        return False, str(exc)


status = check_qlib_status()
col1, col2, col3 = st.columns(3)
col1.metric("Qlib 安装", "✅" if status.get("installed") else "❌")
col2.metric("US 数据", "✅" if status.get("us_data") else "❌")
col3.metric("CN 数据", "✅" if status.get("cn_data") else "❌")

if not status.get("installed"):
    st.warning("当前环境未安装 pyqlib。请先安装后再运行挖掘。")

with st.expander("运行挖掘任务", expanded=False):
    run_market = st.selectbox("市场", ["US", "CN"], index=0)
    run_segment = st.selectbox("市值分层", ["ALL", "LARGE", "MID", "SMALL"], index=0)
    run_days = st.slider("回溯天数", min_value=180, max_value=1460, value=730, step=30)
    topk_grid = st.text_input("TopK 网格", value="5,8,10,15")
    drop_grid = st.text_input("N_drop 网格", value="1,2,3")
    run_batch = st.checkbox("批量跑分层对比（仅 US）", value=True)

    if st.button("开始挖掘", type="primary"):
        cmd = [
            sys.executable,
            "scripts/run_qlib_mining.py",
            "--market",
            run_market,
            "--segment",
            run_segment,
            "--days",
            str(run_days),
            "--topk-grid",
            topk_grid,
            "--drop-grid",
            drop_grid,
        ]
        if run_batch and run_market == "US":
            cmd.append("--run-segment-batch")

        with st.spinner("运行中，可能需要几分钟..."):
            ok, output = _run_command(cmd)

        if ok:
            st.success("挖掘完成")
        else:
            st.error("挖掘失败")
        st.code(output if output else "(无输出)")

market = st.radio("查看市场", ["US", "CN"], horizontal=True)
out_dir = _model_dir(market)

summary = _read_json_if_exists(out_dir / "qlib_mining_summary_latest.json")
factor_df = _read_csv_if_exists(out_dir / "factor_mining_latest.csv")
strategy_df = _read_csv_if_exists(out_dir / "strategy_mining_latest.csv")
segment_df = _read_csv_if_exists(out_dir / "segment_strategy_compare_latest.csv")

st.markdown("---")
st.subheader("最新结果概览")
if summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("市场", str(summary.get("market", "-")))
    c2.metric("分层", str(summary.get("segment", "-")))
    c3.metric("因子数", int(summary.get("factor_rows", 0)))
    c4.metric("策略组合数", int(summary.get("strategy_rows", 0)))

    top_strategy = (summary.get("top_strategies") or [{}])[0]
    if top_strategy:
        st.info(
            "最佳策略: "
            f"topk={top_strategy.get('topk')} n_drop={top_strategy.get('n_drop')} "
            f"ann={float(top_strategy.get('ann_return', 0.0)):.2%} "
            f"sharpe={float(top_strategy.get('sharpe', 0.0)):.2f}"
        )
else:
    st.warning(f"未找到结果文件：{out_dir}")

st.markdown("---")
left, right = st.columns(2)

with left:
    st.subheader("因子排名（Top 30）")
    if factor_df.empty:
        st.caption("暂无数据")
    else:
        show_factor = factor_df.head(30).copy()
        st.dataframe(show_factor, width='stretch')

        chart_df = show_factor.head(15).copy()
        fig = px.bar(
            chart_df,
            x="factor",
            y="score",
            color="abs_ic",
            title="Top 因子综合分",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("策略排名（Top 30）")
    if strategy_df.empty:
        st.caption("暂无数据")
    else:
        show_strategy = strategy_df.head(30).copy()
        st.dataframe(show_strategy, width='stretch')

        chart_df = show_strategy.head(15).copy()
        fig = px.scatter(
            chart_df,
            x="max_drawdown",
            y="ann_return",
            size="score",
            color="sharpe",
            hover_data=["topk", "n_drop", "turnover"],
            title="策略收益-回撤分布",
        )
        st.plotly_chart(fig, width='stretch')

st.markdown("---")
st.subheader("市值分层策略对比")
if segment_df.empty:
    st.caption("暂无分层对比数据。运行任务时勾选“批量跑分层对比（仅 US）”。")
else:
    st.dataframe(segment_df, width='stretch')

    fig1 = px.bar(segment_df, x="segment", y="best_ann_return", color="segment", title="分层最佳策略年化收益")
    fig2 = px.bar(segment_df, x="segment", y="best_sharpe", color="segment", title="分层最佳策略 Sharpe")
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig1, width='stretch')
    c2.plotly_chart(fig2, width='stretch')

st.markdown("---")
st.caption(f"最后刷新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
