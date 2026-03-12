import streamlit as st
import pandas as pd
import glob
import os
import sys
import json
import socket
import subprocess
import urllib.error
import urllib.request
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict

# 添加当前目录到路径，以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from chart_utils import create_candlestick_chart, create_candlestick_chart_dynamic, analyze_chip_flow, create_chip_flow_chart, create_chip_change_chart, quick_chip_analysis
from data_fetcher import get_us_stock_data as fetch_data_from_polygon, get_ticker_details, get_stock_data, get_cn_stock_data
from components.stock_detail import render_unified_stock_detail
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_adx_series
from backtester import SimpleBacktester
from db.database import (
    query_scan_results, get_scanned_dates, get_scan_date_counts, get_db_stats, 
    get_stock_history, init_db, get_scan_job, get_stock_info_batch,
    get_first_scan_dates, USE_SUPABASE, SUPABASE_LAYER_AVAILABLE
)

# 设置页面配置
st.set_page_config(
    page_title="Coral Creek V3.0 - 智能量化系统",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 加载自定义 CSS ---
# --- 加载自定义 CSS ---
def load_custom_css():
    """加载自定义 CSS 样式 (含移动端优化)"""
    
    # 基础样式
    base_css = """
    <style>
        /* 移动端响应式优化 */
        @media (max-width: 768px) {
            /* 调整 tab 字体大小 */
            .stTabs [data-baseweb="tab"] {
                font-size: 0.85em;
                padding: 4px 8px;
                min-width: auto;
            }
            /* 调整指标卡片内边距 */
            div[data-testid="metric-container"] {
                padding: 4px;
                min-height: auto;
            }
            div[data-testid="metric-container"] label {
                font-size: 0.8em;
            }
            div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
                font-size: 1.1em;
            }
            /* 侧边栏调整 */
            section[data-testid="stSidebar"] {
                width: 250px !important;
            }
            /* 按钮间距 */
            .stButton button {
                padding: 0.25rem 0.5rem;
            }
        }
        
        /* 浮动操作栏样式优化 */
        .floating-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: #1E1E1E;
            z-index: 999;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
            border-top: 1px solid #333;
        }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

# 应用自定义样式
load_custom_css()

# --- 环境变量适配 ---
# 将 Streamlit Secrets 注入环境变量
def inject_secrets():
    """将 Streamlit Secrets 注入到环境变量"""
    try:
        if hasattr(st, "secrets"):
            # 遍历所有 secrets
            for key in st.secrets:
                value = st.secrets[key]
                # 只注入字符串值
                if isinstance(value, str):
                    if key not in os.environ or not os.environ[key]:
                        os.environ[key] = value
                        print(f"✅ Injected secret: {key}")
                # 支持分组配置（如 [alpaca]）
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str):
                            env_key = f"{key}_{sub_key}".upper()
                            if env_key not in os.environ or not os.environ[env_key]:
                                os.environ[env_key] = sub_value
            
            # 特别检查 Supabase
            if 'SUPABASE_URL' in os.environ:
                print(f"✅ SUPABASE_URL: {os.environ['SUPABASE_URL'][:30]}...")
            else:
                print("⚠️ SUPABASE_URL not found in secrets")
    except Exception as e:
        print(f"⚠️ Secrets injection error: {e}")

inject_secrets()


def _get_global_paper_account_name() -> str:
    """获取全局模拟盘子账户名"""
    return st.session_state.get("global_paper_account_name", "default")


def _set_global_paper_account_name(name: str):
    """设置全局模拟盘子账户名"""
    st.session_state["global_paper_account_name"] = (name or "default")


def _set_active_market(market: str):
    """记录当前页面市场上下文，用于统一控制 US/CN 交易入口"""
    if market in ("US", "CN"):
        st.session_state["active_market"] = market


def _get_active_market(default: str = "US") -> str:
    """读取当前页面市场上下文"""
    market = st.session_state.get("active_market", default)
    return market if market in ("US", "CN") else default


# --- 数据缓存层 (Performance Optimization) ---
# 全局缓存高频数据查询，避免每次交互都重新加载

@st.cache_data(ttl=300, show_spinner=False)
def _cached_scan_results(scan_date, market, limit=1000):
    """缓存扫描结果 (5分钟TTL)"""
    return query_scan_results(scan_date=scan_date, market=market, limit=limit)

@st.cache_data(ttl=600, show_spinner=False)
def _cached_stock_data(symbol, market='US', days=365):
    """缓存股票历史数据 (10分钟TTL)"""
    return get_stock_data(symbol, market=market, days=days)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_scanned_dates(market=None):
    """缓存可用扫描日期列表 (5分钟TTL)"""
    return get_scanned_dates(market=market)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_scan_date_counts(market=None, limit=30):
    """缓存每日扫描计数 (5分钟TTL) — 轻量 GROUP BY 查询"""
    return get_scan_date_counts(market=market, limit=limit)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_db_stats():
    """缓存数据库统计信息 (1小时TTL)"""
    return get_db_stats()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_theme_radar(market, top_themes, leaders_per_theme, include_social, latest_date):
    """缓存主题雷达，减少重复计算（15分钟TTL）"""
    from services.theme_radar_service import build_theme_radar
    results = _cached_scan_results(scan_date=latest_date, market=market, limit=500)
    scan_df = pd.DataFrame(results) if results else pd.DataFrame()
    return build_theme_radar(
        market=market,
        top_themes=top_themes,
        leaders_per_theme=leaders_per_theme,
        scan_df=scan_df,
        include_social=include_social,
    )


def render_data_source_status_bar():
    """顶部数据源状态条：显示当前数据源模式与 US/CN 最新扫描日期"""
    try:
        us_dates = _cached_scanned_dates(market="US")
        cn_dates = _cached_scanned_dates(market="CN")
        us_latest = us_dates[0] if us_dates else "暂无"
        cn_latest = cn_dates[0] if cn_dates else "暂无"
        source_mode = "Supabase+SQLite容错" if (USE_SUPABASE and SUPABASE_LAYER_AVAILABLE) else "SQLite本地"
        now_txt = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.caption(f"🧭 数据源: {source_mode} | US最新: {us_latest} | CN最新: {cn_latest} | 更新时间: {now_txt}")
    except Exception as e:
        st.caption(f"🧭 数据源状态读取失败: {e}")


@st.cache_data(ttl=300, show_spinner=False)
def _get_action_health_rows():
    """从 git 提交中估算核心 Action 健康状态"""
    workflows = [
        {"name": "Daily Stock Scan (US)", "pattern": "📊 Auto-update: Scan results for"},
        {"name": "Daily Stock Scan (CN)", "pattern": "📊 Auto-update: CN A-Share scan results for"},
        {"name": "Default Baskets Auto Execute", "pattern": "🤖 Auto basket execution update"},
        {"name": "ML Model Training", "pattern": "🧠 Auto-update: ML models retrained"},
        {"name": "Social KOL Scan", "pattern": "📡 Auto-update: social KOL scan"},
    ]
    rows = []
    try:
        proc = subprocess.run(
            [
                "git", "log",
                "--pretty=format:%aI|%aN|%aE|%s",
                "-n", "500",
            ],
            cwd=current_dir,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        lines = (proc.stdout or "").splitlines()
        now = datetime.now().astimezone()
        for wf in workflows:
            latest_ts = None
            latest_msg = ""
            fallback_ts = None
            fallback_msg = ""
            for line in lines:
                if "|" not in line:
                    continue
                parts = line.split("|", 3)
                if len(parts) != 4:
                    continue
                ts_txt, author_name, author_email, msg = parts
                is_actions_bot = (
                    "github-actions[bot]" in (author_name or "").lower()
                    or "github-actions[bot]" in (author_email or "").lower()
                )
                if wf["pattern"] in msg and fallback_ts is None:
                    fallback_msg = msg.strip()
                    try:
                        fallback_ts = datetime.fromisoformat(ts_txt.strip())
                    except Exception:
                        fallback_ts = None
                if is_actions_bot and wf["pattern"] in msg:
                    latest_msg = msg.strip()
                    try:
                        latest_ts = datetime.fromisoformat(ts_txt.strip())
                    except Exception:
                        latest_ts = None
                    break
            if latest_ts is None and fallback_ts is not None:
                latest_ts = fallback_ts
                latest_msg = fallback_msg
            if latest_ts:
                lag_h = (now - latest_ts).total_seconds() / 3600.0
                status = "🟢 正常" if lag_h <= 36 else ("🟡 偏久" if lag_h <= 72 else "🔴 需检查")
                rows.append({
                    "任务": wf["name"],
                    "状态": status,
                    "最近提交时间": latest_ts.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "延迟(小时)": round(lag_h, 1),
                    "最近信息": latest_msg,
                })
            else:
                rows.append({
                    "任务": wf["name"],
                    "状态": "⚪ 无记录",
                    "最近提交时间": "-",
                    "延迟(小时)": None,
                    "最近信息": "-",
                })
    except Exception as e:
        rows = [{
            "任务": "Action Health",
            "状态": "⚠️ 读取失败",
            "最近提交时间": "-",
            "延迟(小时)": None,
            "最近信息": str(e),
        }]
    # 强制清洗列类型，避免 Arrow 因历史缓存/混型数据报错
    for r in rows:
        v = r.get("延迟(小时)")
        if v in ("-", "", "None"):
            r["延迟(小时)"] = None
        else:
            try:
                r["延迟(小时)"] = float(v) if v is not None else None
            except Exception:
                r["延迟(小时)"] = None
    return rows


def render_action_health_panel():
    """Action 健康总览面板"""
    with st.expander("🛠️ Action 健康总览", expanded=False):
        rows = _get_action_health_rows()
        if rows:
            df = pd.DataFrame(rows)
            if "延迟(小时)" in df.columns:
                df["延迟(小时)"] = pd.to_numeric(
                    df["延迟(小时)"].replace({"-": None, "": None}),
                    errors="coerce",
                )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无可展示的 Action 状态")


def _run_network_diagnostics():
    """基础网络连通性诊断（DNS + HTTP）"""
    targets = [
        ("Polygon", "api.polygon.io", "https://api.polygon.io"),
        ("Reddit", "www.reddit.com", "https://www.reddit.com"),
        ("X", "x.com", "https://x.com"),
    ]
    rows = []
    for name, host, url in targets:
        dns_ok = False
        http_ok = False
        dns_msg = ""
        http_msg = ""
        try:
            socket.gethostbyname(host)
            dns_ok = True
            dns_msg = "OK"
        except Exception as e:
            dns_msg = str(e)[:120]

        if dns_ok:
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=6) as resp:
                    code = getattr(resp, "status", 200)
                http_ok = 200 <= int(code) < 500
                http_msg = f"HTTP {code}"
            except urllib.error.HTTPError as e:
                code = int(getattr(e, "code", 0) or 0)
                # 403/404 说明目标可达，只是被拒绝或路径不对
                http_ok = code in (401, 403, 404, 429)
                if code == 404:
                    http_msg = "HTTP 404 (reachable, endpoint not found)"
                elif code == 403:
                    http_msg = "HTTP 403 (reachable, blocked/forbidden)"
                elif code == 429:
                    http_msg = "HTTP 429 (reachable, rate limited)"
                else:
                    http_msg = f"HTTP {code}"
            except Exception as e:
                http_msg = str(e)[:120]
        else:
            http_msg = "skip (dns failed)"

        rows.append({
            "服务": name,
            "主机": host,
            "DNS": "✅" if dns_ok else "❌",
            "HTTP": "✅" if http_ok else "❌",
            "详情": f"DNS: {dns_msg} | HTTP: {http_msg}",
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def _evaluate_ml_vs_baseline(
    market: str,
    days_back: int,
    topk_per_day: int,
    exit_rule: str,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_hold_days: int,
    max_eval_rows: int = 600,
):
    """同口径评估: 基线TopK(按BLUE) vs 排序模型TopK"""
    from services.candidate_tracking_service import get_candidate_tracking_rows, evaluate_exit_rule
    from ml.smart_picker import SmartPicker
    from data_fetcher import get_stock_data

    rows = get_candidate_tracking_rows(market=market, days_back=days_back) or []
    if not rows:
        return {
            "ok": False,
            "reason": "no_tracking_rows",
            "base": {"sample": 0},
            "ml": {"sample": 0},
        }

    rows = rows[: int(max_eval_rows)]
    picker = SmartPicker(market=market, horizon="short")

    per_day = {}
    for r in rows:
        d = str(r.get("signal_date") or "")
        if d:
            per_day.setdefault(d, []).append(r)

    baseline_rows = []
    ml_rows = []
    ranked_cnt = 0
    fallback_cnt = 0

    for _, day_rows in per_day.items():
        base_sorted = sorted(day_rows, key=lambda x: float(x.get("blue_daily") or 0), reverse=True)
        baseline_rows.extend(base_sorted[: int(topk_per_day)])

        scored = []
        for r in day_rows:
            sym = str(r.get("symbol") or "").strip().upper()
            sig_date = str(r.get("signal_date") or "")
            score = None
            if sym and sig_date:
                try:
                    hist = get_stock_data(sym, market=market, days=420)
                    if hist is not None and not hist.empty:
                        hist2 = hist[hist.index <= pd.to_datetime(sig_date)]
                        if hist2 is not None and len(hist2) >= 80:
                            signal_series = pd.Series({
                                "symbol": sym,
                                "price": float(r.get("signal_price") or r.get("current_price") or 0.0),
                                "blue_daily": float(r.get("blue_daily") or 0.0),
                                "blue_weekly": float(r.get("blue_weekly") or 0.0),
                                "blue_monthly": float(r.get("blue_monthly") or 0.0),
                                "is_heima": bool(r.get("heima_daily") or r.get("heima_weekly") or r.get("heima_monthly")),
                            })
                            rank_map = picker._rank_score(signal_series, hist2)
                            w = picker.rank_weights or {"short": 0.55, "medium": 0.30, "long": 0.15}
                            score = (
                                float(w.get("short", 0.0)) * float(rank_map.get("short", 0.0))
                                + float(w.get("medium", 0.0)) * float(rank_map.get("medium", 0.0))
                                + float(w.get("long", 0.0)) * float(rank_map.get("long", 0.0))
                            )
                            ranked_cnt += 1
                except Exception:
                    score = None

            if score is None:
                score = float(r.get("blue_daily") or 0.0) * 0.7 + float(r.get("blue_weekly") or 0.0) * 0.3
                fallback_cnt += 1
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        ml_rows.extend([x[1] for x in scored[: int(topk_per_day)]])

    base_eval = evaluate_exit_rule(
        baseline_rows,
        rule_name=exit_rule,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_hold_days=max_hold_days,
        max_rows=max_eval_rows,
    )
    ml_eval = evaluate_exit_rule(
        ml_rows,
        rule_name=exit_rule,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_hold_days=max_hold_days,
        max_rows=max_eval_rows,
    )

    return {
        "ok": True,
        "base": base_eval,
        "ml": ml_eval,
        "meta": {
            "days_back": days_back,
            "topk_per_day": topk_per_day,
            "input_rows": len(rows),
            "n_days": len(per_day),
            "ranked_cnt": ranked_cnt,
            "fallback_cnt": fallback_cnt,
        },
    }


@st.cache_data(ttl=900, show_spinner=False)
def _analyze_extreme_lift(
    market: str,
    days_back: int = 360,
    exit_rule: str = "fixed_10d",
    take_profit_pct: float = 10.0,
    stop_loss_pct: float = 6.0,
    max_hold_days: int = 20,
    max_rows: int = 1500,
    schema_ver: int = 2,
) -> Dict:
    """
    极致条件 Lift 分析:
    例如 日BLUE>=200 + 周BLUE>=200 + 黑马/掘地 的胜率提升。
    """
    from services.candidate_tracking_service import get_candidate_tracking_rows, evaluate_exit_rule
    from db.database import get_connection

    rows = get_candidate_tracking_rows(market=market, days_back=days_back) or []
    if not rows:
        return {"ok": False, "reason": "no_tracking_rows", "table": [], "target_col": None}

    # 统一口径：Lift 直接使用同一平仓规则下的交易收益，不再混用 pnl_d*
    eval_ret = evaluate_exit_rule(
        rows=rows,
        rule_name=exit_rule,
        take_profit_pct=float(take_profit_pct),
        stop_loss_pct=float(stop_loss_pct),
        max_hold_days=int(max_hold_days),
        max_rows=int(max_rows),
    )
    details = list(eval_ret.get("details") or [])
    if not details:
        return {"ok": False, "reason": "no_exit_details", "table": [], "target_col": "exit_return_pct"}

    row_map = {}
    for r in rows:
        key = (
            str(r.get("symbol") or "").upper(),
            str(r.get("signal_date") or ""),
            str(r.get("market") or market),
        )
        row_map[key] = r

    # 兼容历史数据：若旧 candidate_tracking 尚未持久化 juedi 字段，则从 scan_results 回补
    min_date = min(str(r.get("signal_date") or "9999-12-31") for r in rows)
    juedi_map = {}
    try:
        conn = get_connection()
        q = """
            SELECT symbol, scan_date, COALESCE(is_juedi, 0) AS is_juedi
            FROM scan_results
            WHERE market = ? AND scan_date >= ?
        """
        dfj = pd.read_sql_query(q, conn, params=(market, min_date))
        conn.close()
        for _, x in dfj.iterrows():
            key = (str(x.get("symbol") or "").upper(), str(x.get("scan_date") or ""))
            juedi_map[key] = int(float(x.get("is_juedi") or 0))
    except Exception:
        juedi_map = {}

    def _to_float(v, d=0.0):
        try:
            if v is None:
                return float(d)
            return float(v)
        except Exception:
            return float(d)

    enriched = []
    for d in details:
        sym = str(d.get("symbol") or "").upper()
        dt = str(d.get("signal_date") or "")
        mk = str(d.get("market") or market)
        r = row_map.get((sym, dt, mk)) or row_map.get((sym, dt, market)) or {}
        ret = _to_float(d.get("exit_return_pct"), np.nan)
        if np.isnan(ret):
            continue
        enriched.append(
            {
                "symbol": sym,
                "signal_date": dt,
                "market": mk,
                "signal_price": _to_float(r.get("signal_price"), np.nan),
                "current_price": _to_float(r.get("current_price"), np.nan),
                "exit_day": int(_to_float(d.get("exit_day"), 0)),
                "ret": ret,
                "blue_daily": _to_float(r.get("blue_daily")),
                "blue_weekly": _to_float(r.get("blue_weekly")),
                "blue_monthly": _to_float(r.get("blue_monthly")),
                "heima_daily": bool(r.get("heima_daily")),
                "heima_weekly": bool(r.get("heima_weekly")),
                "heima_monthly": bool(r.get("heima_monthly")),
                "juedi_daily": bool(r.get("juedi_daily")) or bool(juedi_map.get((sym, dt), 0)),
                "juedi_weekly": bool(r.get("juedi_weekly")),
                "juedi_monthly": bool(r.get("juedi_monthly")),
            }
        )

    if not enriched:
        return {"ok": False, "reason": "no_valid_returns", "table": [], "target_col": "exit_return_pct"}

    def _signal_text(x: Dict) -> str:
        labels = []
        if x.get("heima_daily"):
            labels.append("日黑马")
        if x.get("heima_weekly"):
            labels.append("周黑马")
        if x.get("heima_monthly"):
            labels.append("月黑马")
        if x.get("juedi_daily"):
            labels.append("日掘地")
        if x.get("juedi_weekly"):
            labels.append("周掘地")
        if x.get("juedi_monthly"):
            labels.append("月掘地")
        return "、".join(labels) if labels else "无"

    def _eval(name, fn):
        picked = [x for x in enriched if fn(x)]
        n = len(picked)
        if n == 0:
            return {"组合": name, "样本": 0, "胜率(%)": None, "均收(%)": None, "相对基线提升(%)": None, "__details": []}
        wr = float(sum(1 for x in picked if x["ret"] > 0) / n * 100.0)
        avg = float(np.mean([x["ret"] for x in picked]))
        detail_rows = []
        for x in picked:
            detail_rows.append(
                {
                    "symbol": x.get("symbol"),
                    "signal_date": x.get("signal_date"),
                    "signal_price": round(_to_float(x.get("signal_price"), np.nan), 4) if np.isfinite(_to_float(x.get("signal_price"), np.nan)) else None,
                    "current_price": round(_to_float(x.get("current_price"), np.nan), 4) if np.isfinite(_to_float(x.get("current_price"), np.nan)) else None,
                    "exit_day": int(x.get("exit_day") or 0),
                    "exit_return_pct": round(_to_float(x.get("ret"), 0.0), 2),
                    "is_win": 1 if _to_float(x.get("ret"), 0.0) > 0 else 0,
                    "signal_tags": _signal_text(x),
                    "blue_daily": round(_to_float(x.get("blue_daily"), 0.0), 1),
                    "blue_weekly": round(_to_float(x.get("blue_weekly"), 0.0), 1),
                    "blue_monthly": round(_to_float(x.get("blue_monthly"), 0.0), 1),
                }
            )
        detail_rows.sort(key=lambda z: str(z.get("signal_date") or ""), reverse=True)
        return {"组合": name, "样本": n, "胜率(%)": round(wr, 1), "均收(%)": round(avg, 2), "__details": detail_rows}

    baseline = _eval("基线(全部候选)", lambda x: True)
    base_wr = float(baseline.get("胜率(%)") or 0.0)

    def _base_200(x):
        return x["blue_daily"] >= 200 and x["blue_weekly"] >= 200

    def _any_heima(x):
        return x["heima_daily"] or x["heima_weekly"] or x["heima_monthly"]

    def _any_juedi(x):
        return x["juedi_daily"] or x["juedi_weekly"] or x["juedi_monthly"]

    def _heima_and_juedi_cross_cycle(x):
        """
        黑马/掘地在同一周期互斥，必须跨周期同时出现才算有效组合
        """
        return (
            (x["heima_daily"] and (x["juedi_weekly"] or x["juedi_monthly"]))
            or (x["heima_weekly"] and (x["juedi_daily"] or x["juedi_monthly"]))
            or (x["heima_monthly"] and (x["juedi_daily"] or x["juedi_weekly"]))
        )

    combos = [
        _eval("日200+周200", lambda x: _base_200(x)),
        _eval(
            "日200+周200+日/周黑马",
            lambda x: _base_200(x) and (x["heima_daily"] or x["heima_weekly"]),
        ),
        _eval(
            "日200+周200+日黑马+周黑马(同时)",
            lambda x: _base_200(x) and x["heima_daily"] and x["heima_weekly"],
        ),
        _eval(
            "日200+周200+任一黑马",
            lambda x: _base_200(x) and _any_heima(x),
        ),
        _eval(
            "日200+周200+日/周黑马+跨周期掘地",
            lambda x: _base_200(x) and (x["heima_daily"] or x["heima_weekly"]) and _heima_and_juedi_cross_cycle(x),
        ),
        _eval(
            "日200+周200+日黑马+周黑马+月掘地(跨周期)",
            lambda x: _base_200(x) and x["heima_daily"] and x["heima_weekly"] and x["juedi_monthly"],
        ),
        _eval(
            "日200+周200+月200",
            lambda x: _base_200(x) and x["blue_monthly"] >= 200,
        ),
        _eval(
            "日200+周200+月200+日+周+月黑马(同时)",
            lambda x: _base_200(x) and x["blue_monthly"] >= 200 and x["heima_daily"] and x["heima_weekly"] and x["heima_monthly"],
        ),
        _eval(
            "日200+周200+月200+任一黑马",
            lambda x: _base_200(x) and x["blue_monthly"] >= 200 and _any_heima(x),
        ),
        _eval(
            "日200+周200+月200+任一黑马+任一掘地",
            lambda x: _base_200(x) and x["blue_monthly"] >= 200 and _heima_and_juedi_cross_cycle(x),
        ),
    ]

    table_raw = [baseline] + combos
    combo_details = {str(x.get("组合")): list(x.get("__details") or []) for x in table_raw}
    table = [{k: v for k, v in x.items() if k != "__details"} for x in table_raw]
    base_n = int(baseline.get("样本") or 0)
    for r in table:
        wr = r.get("胜率(%)")
        if wr is None:
            r["相对基线提升(%)"] = None
        else:
            r["相对基线提升(%)"] = round(float(wr) - base_wr, 1)
        cur_n = int(r.get("样本") or 0)
        r["占基线比例(%)"] = round(cur_n / base_n * 100.0, 2) if base_n > 0 else None
        r["样本可靠性"] = "低(样本<20)" if (r.get("样本") or 0) < 20 else "正常"

    return {
        "ok": True,
        "target_col": "exit_return_pct",
        "table": table,
        "combo_details": combo_details,
        "base_sample": int(baseline.get("样本") or 0),
        "rule_name": exit_rule,
        "schema_ver": int(schema_ver),
    }


def _primary_strategy_bucket(tags: list) -> str:
    t = set(tags or [])
    if {"DAY_BLUE", "WEEK_BLUE", "MONTH_BLUE"}.issubset(t):
        return "三线共振"
    if {"DUOKONGWANG_BUY", "DAY_BLUE"}.issubset(t):
        return "多空王+蓝线"
    if {"DAY_BLUE", "WEEK_BLUE"}.issubset(t):
        return "日周共振"
    if {"DAY_HEIMA", "WEEK_HEIMA", "MONTH_HEIMA"} & t:
        return "黑马系"
    if {"CHIP_BREAKOUT", "CHIP_DENSE"} & t:
        return "筹码系"
    if "DUOKONGWANG_BUY" in t:
        return "多空王买点"
    if "DAY_BLUE" in t:
        return "日线趋势"
    if "WEEK_BLUE" in t:
        return "周线趋势"
    if "MONTH_BLUE" in t:
        return "月线趋势"
    return "其他"


def _combo_bucket(tags: list) -> str:
    t = set(tags or [])
    if {"DAY_BLUE", "WEEK_BLUE", "MONTH_BLUE"}.issubset(t):
        return "三线共振"
    if {"DAY_BLUE", "WEEK_BLUE"}.issubset(t):
        return "日周共振"
    if {"DAY_HEIMA", "WEEK_HEIMA"}.issubset(t):
        return "日周黑马同现"
    if {"DAY_HEIMA", "WEEK_HEIMA", "MONTH_HEIMA"} & t:
        return "任一黑马"
    if "DUOKONGWANG_BUY" in t and "DAY_BLUE" in t:
        return "多空王+日蓝"
    if "DUOKONGWANG_BUY" in t:
        return "多空王买点"
    if {"CHIP_BREAKOUT", "CHIP_DENSE"} & t:
        return "筹码结构"
    if "DAY_BLUE" in t:
        return "日蓝"
    if "WEEK_BLUE" in t:
        return "周蓝"
    return "其他"


def _combo_bucket_full(tags: list) -> str:
    """
    全量标签组合（不压缩），用于查看真实组合分布。
    例如: 日蓝+周蓝+月蓝+日黑马+筹码密集
    """
    t = {str(x).strip().upper() for x in (tags or []) if str(x).strip()}
    if not t:
        return "无标签"
    order = [
        ("DAY_BLUE", "日蓝"),
        ("WEEK_BLUE", "周蓝"),
        ("MONTH_BLUE", "月蓝"),
        ("DAY_HEIMA", "日黑马"),
        ("WEEK_HEIMA", "周黑马"),
        ("MONTH_HEIMA", "月黑马"),
        ("DAY_JUEDI", "日掘地"),
        ("WEEK_JUEDI", "周掘地"),
        ("MONTH_JUEDI", "月掘地"),
        ("CHIP_BREAKOUT", "筹码突破"),
        ("CHIP_DENSE", "筹码密集"),
        ("CHIP_OVERHANG", "筹码套牢"),
        ("DUOKONGWANG_BUY", "多空王买点"),
    ]
    labels = [cn for k, cn in order if k in t]
    if not labels:
        return "其他标签"
    return "+".join(labels)


def _strategy_bucket_full(tags: list) -> str:
    """
    策略明细口径：当前直接使用全量标签组合，避免被粗分类折叠。
    """
    return _combo_bucket_full(tags)


def _effective_tags_for_fact(row: dict) -> list:
    """
    兼容历史数据: 若 signal_tags_list 为空，则基于字段实时重建标签，
    避免策略组合被大量“无标签”样本压扁。
    """
    tags = list(row.get("signal_tags_list") or [])
    if tags:
        return tags

    out = []
    try:
        if float(row.get("blue_daily") or 0) >= 100:
            out.append("DAY_BLUE")
        if float(row.get("blue_weekly") or 0) >= 80:
            out.append("WEEK_BLUE")
        if float(row.get("blue_monthly") or 0) >= 60:
            out.append("MONTH_BLUE")
    except Exception:
        pass

    if bool(row.get("heima_daily")):
        out.append("DAY_HEIMA")
    if bool(row.get("heima_weekly")):
        out.append("WEEK_HEIMA")
    if bool(row.get("heima_monthly")):
        out.append("MONTH_HEIMA")
    if bool(row.get("juedi_daily")):
        out.append("DAY_JUEDI")
    if bool(row.get("juedi_weekly")):
        out.append("WEEK_JUEDI")
    if bool(row.get("juedi_monthly")):
        out.append("MONTH_JUEDI")
    if bool(row.get("duokongwang_buy")):
        out.append("DUOKONGWANG_BUY")

    try:
        pr = float(row.get("profit_ratio") or 0)
        if pr >= 0.7:
            out.append("CHIP_DENSE")
        if pr >= 0.9:
            out.append("CHIP_BREAKOUT")
        if 0 < pr <= 0.3:
            out.append("CHIP_OVERHANG")
    except Exception:
        pass

    return out


def _effective_tags_for_fact_relaxed(row: dict) -> list:
    """
    宽松标签口径（独立统计，不并入严格组合）：
    - Blue 阈值放宽到 50/50/50
    - 筹码阈值放宽到 0.6 / 0.85 / 0.35
    """
    out = []
    try:
        if float(row.get("blue_daily") or 0) >= 50:
            out.append("DAY_BLUE")
        if float(row.get("blue_weekly") or 0) >= 50:
            out.append("WEEK_BLUE")
        if float(row.get("blue_monthly") or 0) >= 50:
            out.append("MONTH_BLUE")
    except Exception:
        pass

    if bool(row.get("heima_daily")):
        out.append("DAY_HEIMA")
    if bool(row.get("heima_weekly")):
        out.append("WEEK_HEIMA")
    if bool(row.get("heima_monthly")):
        out.append("MONTH_HEIMA")
    if bool(row.get("juedi_daily")):
        out.append("DAY_JUEDI")
    if bool(row.get("juedi_weekly")):
        out.append("WEEK_JUEDI")
    if bool(row.get("juedi_monthly")):
        out.append("MONTH_JUEDI")
    if bool(row.get("duokongwang_buy")):
        out.append("DUOKONGWANG_BUY")

    try:
        pr = float(row.get("profit_ratio") or 0)
        if pr >= 0.6:
            out.append("CHIP_DENSE")
        if pr >= 0.85:
            out.append("CHIP_BREAKOUT")
        if 0 < pr <= 0.35:
            out.append("CHIP_OVERHANG")
    except Exception:
        pass

    return out


def _build_unified_trade_facts(
    rows: list,
    exit_rule: str,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_hold_days: int,
    max_rows: int = 1500,
) -> pd.DataFrame:
    from services.candidate_tracking_service import evaluate_exit_rule

    if not rows:
        return pd.DataFrame()

    # -------- 优先读预计算表（秒出） --------
    details = []
    try:
        from scripts.precompute_exit_results import get_precomputed_details
        market_val = str(rows[0].get("market") or "US") if rows else "US"
        details = get_precomputed_details(
            market=market_val,
            rule_name=exit_rule,
            tp=float(take_profit_pct),
            sl=float(stop_loss_pct),
            hold=int(max_hold_days),
        )
        if details:
            # 预计算表有数据 → 直接使用，跳过实时计算
            pass
    except Exception:
        details = []

    # -------- 兜底：实时计算（上限 5000，避免卡死） --------
    if not details:
        safe_max = min(int(max_rows), 5000)
        eval_ret = evaluate_exit_rule(
            rows=rows,
            rule_name=exit_rule,
            take_profit_pct=float(take_profit_pct),
            stop_loss_pct=float(stop_loss_pct),
            max_hold_days=int(max_hold_days),
            max_rows=safe_max,
        )
        details = list(eval_ret.get("details") or [])

    if not details:
        return pd.DataFrame()

    row_map = {}
    for r in rows:
        row_map[(str(r.get("symbol") or "").upper(), str(r.get("signal_date") or ""), str(r.get("market") or ""))] = r

    facts = []
    for d in details:
        sym = str(d.get("symbol") or "").upper()
        dt = str(d.get("signal_date") or "")
        mk = str(d.get("market") or "")
        src = row_map.get((sym, dt, mk)) or row_map.get((sym, dt, "")) or {}
        tags = _effective_tags_for_fact(src)
        tags_relaxed = _effective_tags_for_fact_relaxed(src)
        ret = float(d.get("exit_return_pct") or 0.0)
        facts.append(
            {
                "symbol": sym,
                "signal_date": dt,
                "market": mk,
                "ret": ret,
                "win": 1 if ret > 0 else 0,
                "strategy_bucket": _primary_strategy_bucket(tags),
                "combo_bucket": _combo_bucket(tags),
                "strategy_bucket_full": _strategy_bucket_full(tags),
                "combo_bucket_full": _combo_bucket_full(tags),
                "strategy_bucket_relaxed": _strategy_bucket_full(tags_relaxed),
                "combo_bucket_relaxed": _combo_bucket_full(tags_relaxed),
                "cap_category": str(src.get("cap_category") or "未知"),
                "industry": str(src.get("industry") or "Unknown"),
            }
        )
    return pd.DataFrame(facts)


# --- 后台调度器 (In-App Scheduler) ---
# 替代 GitHub Actions，直接在应用内运行监控
# 避免支付问题和数据同步问题

@st.cache_resource
def init_scheduler(_scheduler_rev: str = "2026-02-12-r2-fix-kwargs"):
    """初始化并启动后台调度器 (单例模式)"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from scripts.intraday_monitor import monitor_portfolio
        import atexit
        
        # 创建调度器（通过 _scheduler_rev 变更可强制刷新缓存中的旧任务定义）
        scheduler = BackgroundScheduler()
        
        # 防止重复添加
        if scheduler.get_job('intraday_monitor_job'):
            return scheduler
        
        # 定义任务
        def job_function():
            from datetime import datetime
            print(f"📱 盘中监控 - {datetime.now()}")
            try:
                # 统一监控当前持仓（函数签名无 market/run_once）
                monitor_portfolio()
            except Exception as e:
                print(f"⚠️ [Scheduler] Job failed: {e}")
        
        # 添加任务 (每30分钟)
        scheduler.add_job(
            job_function,
            IntervalTrigger(minutes=30),
            id='intraday_monitor_job',
            replace_existing=True,
            name='Intraday Monitor (Every 30min)'
        )
        
        # 启动
        scheduler.start()
        print("✅ [Scheduler] Background scheduler started (Interval: 30min)")
        
        # 退出时关闭
        atexit.register(lambda: scheduler.shutdown())
        
        return scheduler
    except ImportError:
        print("⚠️ [Scheduler] APScheduler not installed. Skipping.")
        return None
    except Exception as e:
        print(f"⚠️ [Scheduler] Failed to start: {e}")
        return None

# 启动调度器
init_scheduler("2026-02-12-r1")

# --- 登录验证 ---

def check_password():
    """角色验证 - Admin 可管理持仓，Guest 只能查看"""
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = None
    
    if st.session_state["user_role"] is None:
        st.markdown("## 🦅 Coral Creek V3.0")
        st.markdown("智能量化扫描系统")
        st.markdown("---")
        
        password = st.text_input("密码", type="password", key="password_input")
        
        if st.button("登录", type="primary"):
            # 获取密码配置
            try:
                admin_password = st.secrets.get("admin_password", "admin2026")
                guest_password = st.secrets.get("guest_password", "coral2026")
            except:
                admin_password = "admin2026"
                guest_password = "coral2026"
            
            if password == admin_password:
                st.session_state["user_role"] = "admin"
                st.success("✅ 欢迎，管理员！")
                st.rerun()
            elif password == guest_password:
                st.session_state["user_role"] = "guest"
                st.success("✅ 欢迎访客！")
                st.rerun()
            elif password:
                st.error("❌ 密码错误")
        
        st.markdown("---")
        st.caption("Admin: 完整功能 | Guest: 只读模式")
        st.stop()

def is_admin():
    """检查当前用户是否为管理员"""
    return st.session_state.get("user_role") == "admin"

check_password()

# --- 侧边栏: Alpaca 持仓 + 系统工具 ---
with st.sidebar:
    # Alpaca 持仓小部件（仅 US 市场）
    try:
        from components.alpaca_widget import render_alpaca_sidebar_widget
        sidebar_market = _get_active_market()
        render_alpaca_sidebar_widget(
            enabled=(sidebar_market == "US"),
            current_market=sidebar_market,
        )
    except ImportError:
        pass  # 组件未安装时静默跳过
    
    st.markdown("---")
    st.caption("🎮 模拟盘")
    try:
        from services.portfolio_service import list_paper_accounts
        sidebar_accounts = list_paper_accounts()
        sidebar_names = [a['account_name'] for a in sidebar_accounts] if sidebar_accounts else ['default']
        current_global_account = _get_global_paper_account_name()
        sidebar_index = sidebar_names.index(current_global_account) if current_global_account in sidebar_names else 0
        selected_global_account = st.selectbox(
            "全局子账户",
            sidebar_names,
            index=sidebar_index,
            key="sidebar_global_paper_account"
        )
        _set_global_paper_account_name(selected_global_account)
    except Exception:
        st.caption("⚠️ 模拟盘子账户加载失败")

    st.markdown("---")

    st.caption("🔧 系统工具")
    if st.button("🔔 发送测试通知", help="点击此按钮测试 Telegram 连接"):
        from scripts.intraday_monitor import send_alert_telegram
        with st.spinner("正在发送测试消息..."):
            success = send_alert_telegram([{
                'type': 'test',
                'level': '🔔',
                'symbol': '从网站发出',
                'message': '这是一条测试消息',
                'footer': '如果您收到此消息，说明网站监控功能正常。'
            }])
            if success:
                st.toast("✅ 测试消息发送成功!", icon="✅")
            else:
                st.error("❌ 发送失败，请检查 Logs")
    
    # Supabase 调试
    if st.button("🔍 检查数据库", help="检查 Supabase 连接和数据"):
        st.write("**环境变量检查:**")
        supabase_url = os.environ.get('SUPABASE_URL', 'NOT SET')
        supabase_key = os.environ.get('SUPABASE_KEY', 'NOT SET')
        st.write(f"- SUPABASE_URL: `{supabase_url[:40] if supabase_url else 'None'}...`")
        st.write(f"- SUPABASE_KEY: `{'SET' if supabase_key and len(supabase_key) > 10 else 'NOT SET'}`")
        
        # 测试连接
        try:
            from db.supabase_db import get_supabase, is_supabase_available
            if is_supabase_available():
                supabase = get_supabase()
                result = supabase.table('scan_results').select('*').limit(5).execute()
                st.success(f"✅ Supabase 连接成功! 获取到 {len(result.data)} 条记录")
                if result.data:
                    # 检查 heima 列是否存在
                    cols = list(result.data[0].keys())
                    heima_cols = [c for c in cols if 'heima' in c.lower()]
                    st.write(f"**heima 相关列**: {heima_cols if heima_cols else '❌ 无'}")
                    st.json(result.data[0])
            else:
                st.error("❌ Supabase 不可用")
        except Exception as e:
            st.error(f"❌ 连接错误: {e}")
    
    # 修复 Supabase 表结构
    if st.button("🔧 修复黑马列", help="添加缺失的 heima_daily/weekly/monthly 列"):
        try:
            from db.supabase_db import get_supabase, is_supabase_available
            if is_supabase_available():
                supabase = get_supabase()
                
                # 检查是否需要添加列
                result = supabase.table('scan_results').select('*').limit(1).execute()
                if result.data:
                    existing_cols = set(result.data[0].keys())
                    needed_cols = ['heima_daily', 'heima_weekly', 'heima_monthly', 
                                   'juedi_daily', 'juedi_weekly', 'juedi_monthly']
                    missing_cols = [c for c in needed_cols if c not in existing_cols]
                    
                    if not missing_cols:
                        st.success("✅ 所有 heima 列已存在，无需修复")
                    else:
                        st.warning(f"缺失列: {missing_cols}")
                        st.info("""
请在 Supabase SQL Editor 中运行:
```sql
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS heima_daily BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS heima_weekly BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS heima_monthly BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS juedi_daily BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS juedi_weekly BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS juedi_monthly BOOLEAN DEFAULT FALSE;
```
                        """)
                else:
                    st.warning("表为空，请先运行扫描")
            else:
                st.error("❌ Supabase 不可用")
        except Exception as e:
            st.error(f"❌ 错误: {e}")

# --- 工具函数 ---

def format_large_number(num):
    """格式化大数字 (B/M/K)"""
    if not num or pd.isna(num):
        return "N/A"
    num = float(num)
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"

def load_scan_results_from_db(scan_date=None, market=None):
    """从数据库加载扫描结果"""
    try:
        # 如果没有指定日期，获取最新日期
        if scan_date is None:
            dates = _cached_scanned_dates()
            if not dates:
                return None, None
            scan_date = dates[0]  # 最新日期
        
        # 查询数据 - 传入 market 参数 (使用缓存)
        results = _cached_scan_results(scan_date=scan_date, market=market)
        if not results:
            return None, scan_date
        
        df = pd.DataFrame(results)
        
        # --- 数据标准化与列名映射 ---
        col_map = {
            'symbol': 'Ticker',
            'blue_daily': 'Day BLUE',
            'blue_weekly': 'Week BLUE',
            'blue_monthly': 'Month BLUE',
            'stop_loss': 'Stop Loss',
            'shares_rec': 'Shares Rec',
            'vp_rating': 'Vol Profile',
            'market_cap': 'Mkt Cap Raw',
            'company_name': 'Name',
            'industry': 'Industry',
            'turnover_m': 'Turnover',
            'price': 'Price',
            'adx': 'ADX',
            'volatility': 'Volatility',
            'is_heima': 'Is_Heima',
            'is_juedi': 'Is_Juedi',
            'heima_daily': 'Heima_Daily',
            'heima_weekly': 'Heima_Weekly',
            'heima_monthly': 'Heima_Monthly',
            'juedi_daily': 'Juedi_Daily',
            'juedi_weekly': 'Juedi_Weekly',
            'juedi_monthly': 'Juedi_Monthly',
            'strat_d_trend': 'Strat_D_Trend',
            'strat_c_resonance': 'Strat_C_Resonance',
            'legacy_signal': 'Legacy_Signal',
            'regime': 'Regime',
            'adaptive_thresh': 'Adaptive_Thresh',
            'profit_ratio': 'Profit_Ratio',
            'wave_phase': 'Wave_Phase',
            'wave_desc': 'Wave_Desc',
            'chan_signal': 'Chan_Signal',
            'chan_desc': 'Chan_Desc',
            'cap_category': 'Cap_Category',
            'risk_reward_score': 'Risk_Reward_Score',
            'scan_date': 'Date'
        }
        df.rename(columns=col_map, inplace=True)
        
        # 转换布尔字段 (SQLite=bytes, Supabase=bool/str)
        def robust_bool_convert(x):
            """健壮的布尔转换，处理所有可能的数据来源"""
            if x is None:
                return False
            if isinstance(x, bool):
                return x
            if isinstance(x, bytes):
                return x == b'\x01'
            if isinstance(x, (int, float)):
                return x == 1
            if isinstance(x, str):
                return x.lower() in ('true', '1', 't', 'yes')
            return False
        
        bool_cols = ['Is_Heima', 'Is_Juedi', 'Heima_Daily', 'Heima_Weekly', 'Heima_Monthly', 
                     'Juedi_Daily', 'Juedi_Weekly', 'Juedi_Monthly', 'Strat_D_Trend', 'Strat_C_Resonance']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].apply(robust_bool_convert)
        
        # 格式化市值
        if 'Mkt Cap Raw' in df.columns:
            df['Mkt Cap'] = pd.to_numeric(df['Mkt Cap Raw'], errors='coerce').fillna(0) / 1_000_000_000
        else:
            df['Mkt Cap'] = 0.0

        # [补丁] A股数据如果市值为0，尝试用 AkShare/yfinance 实时数据补全
        if market == 'CN' and (df['Mkt Cap'] == 0).mean() > 0.5:
            try:
                import streamlit as st
                
                # 缓存一下，避免每次rerun都拉取全市场
                cache_key = f"cn_mkt_cap_{datetime.now().strftime('%Y%m%d_%H')}"
                
                if cache_key in st.session_state:
                    mkt_map = st.session_state[cache_key]
                else:
                    mkt_map = {}
                    
                    # 方法1: 尝试 AkShare
                    try:
                        import akshare as ak
                        spot_df = ak.stock_zh_a_spot_em()
                        mkt_map = dict(zip(spot_df['代码'], spot_df['总市值']))
                    except Exception as e1:
                        print(f"AkShare failed: {e1}")
                        
                        # 方法2: 尝试 yfinance 批量获取 (只取前30个)
                        try:
                            import yfinance as yf
                            tickers = df['Ticker'].head(30).tolist()
                            yf_symbols = []
                            for t in tickers:
                                code = t.split('.')[0]
                                suffix = '.SS' if t.endswith('.SH') else '.SZ'
                                yf_symbols.append(code + suffix)
                            
                            objs = yf.Tickers(' '.join(yf_symbols))
                            for t, yf_t in zip(tickers, yf_symbols):
                                try:
                                    code = t.split('.')[0]
                                    mc = objs.tickers[yf_t].fast_info.get('marketCap', 0)
                                    if mc:
                                        mkt_map[code] = mc
                                except:
                                    pass
                        except Exception as e2:
                            print(f"yfinance CN failed: {e2}")
                    
                    st.session_state[cache_key] = mkt_map
                
                if mkt_map:
                    def fill_cn_cap(row):
                        if row['Mkt Cap'] > 0: 
                            return row['Mkt Cap']
                        code = row['Ticker'].split('.')[0]
                        cap = mkt_map.get(code, 0)
                        if cap and cap > 0:
                            return cap / 1_000_000_000
                        return 0
                    
                    df['Mkt Cap'] = df.apply(fill_cn_cap, axis=1)
                    
                    # 重新计算 Cap Category
                    def update_category(cap):
                        if cap >= 200: return 'Mega-Cap (超大盘)'
                        elif cap >= 10: return 'Large-Cap (大盘)'
                        elif cap >= 2: return 'Mid-Cap (中盘)'
                        elif cap >= 0.3: return 'Small-Cap (小盘)'
                        return 'Micro-Cap (微盘)'
                    
                    df['Cap_Category'] = df['Mkt Cap'].apply(update_category)
                
            except Exception as e:
                print(f"CN market cap fix failed: {e}")
        
        # [补丁] 美股数据如果市值为0，尝试用 yfinance 和 Polygon 补全
        if market == 'US' and (df['Mkt Cap'] == 0).mean() > 0.5:
            try:
                # 只修复前 30 个，避免加载太慢
                tickers_to_fix = df[df['Mkt Cap'] == 0]['Ticker'].tolist()[:30]
                
                if tickers_to_fix:
                    @st.cache_data(ttl=3600, show_spinner=False)
                    def fetch_us_caps_cached(tickers):
                        caps = {}
                        # 1. 尝试 Yahoo Finance
                        try:
                            import yfinance as yf
                            txt = " ".join(tickers)
                            objs = yf.Tickers(txt)
                            for t in tickers:
                                try:
                                    val = objs.tickers[t].fast_info.market_cap
                                    if val: caps[t] = val / 1_000_000_000
                                except: pass
                        except Exception as ye:
                             print(f"YF Error: {ye}")
                        
                        # 2. 尝试 Polygon (作为补充)
                        try:
                            from data_fetcher import get_ticker_details
                            import time
                            # 只对还没拿到的尝试，且限制数量防止超时
                            missing = [t for t in tickers if t not in caps][:10]
                            for t in missing:
                                try:
                                    det = get_ticker_details(t)
                                    if det and det.get('market_cap'):
                                        caps[t] = det.get('market_cap') / 1_000_000_000
                                    time.sleep(0.25) # 避免限流 (5 calls/min limit for free tier)
                                except: pass
                        except: pass
                        
                        return caps

                    caps_map = fetch_us_caps_cached(tickers_to_fix)
                    
                    def fill_us_cap(row):
                         if row['Mkt Cap'] > 0: return row['Mkt Cap']
                         return caps_map.get(row['Ticker'], 0)
                    
                    df['Mkt Cap'] = df.apply(fill_us_cap, axis=1)

                    def update_category_us(cap):
                        if cap == 0: return 'Unknown'
                        if cap >= 200: return 'Mega-Cap (超大盘)'
                        elif cap >= 10: return 'Large-Cap (大盘)'
                        elif cap >= 2: return 'Mid-Cap (中盘)'
                        elif cap >= 0.3: return 'Small-Cap (小盘)'
                        return 'Micro-Cap (微盘)'
                    df['Cap_Category'] = df['Mkt Cap'].apply(update_category_us)
            except Exception as e:
                print(f"US Cap fix failed: {e}")
        
        # 合成 Strategy 列
        def get_strategy_label(row):
            strategies = []
            if row.get('Strat_D_Trend', False):
                strategies.append('Trend-D')
            if row.get('Strat_C_Resonance', False):
                strategies.append('Resonance-C')
            if not strategies and row.get('Legacy_Signal', False):
                strategies.append('Legacy')
            return " | ".join(strategies) if strategies else "N/A"
            
        df['Strategy'] = df.apply(get_strategy_label, axis=1)
        
        # 合成 Score 列
        def calculate_score(row):
            import math
            score = 0
            blue = row.get('Day BLUE', 0)
            blue = 0 if (blue is None or (isinstance(blue, float) and math.isnan(blue))) else blue
            score += min(blue / 200, 1.0) * 40
            adx = row.get('ADX', 0)
            adx = 0 if (adx is None or (isinstance(adx, float) and math.isnan(adx))) else adx
            score += min(adx / 60, 1.0) * 30
            pr = row.get('Profit_Ratio', 0.5)
            pr = 0.5 if (pr is None or (isinstance(pr, float) and math.isnan(pr))) else pr
            score += pr * 30
            return int(score)
            
        df['Score'] = df.apply(calculate_score, axis=1)
        
        # 类型转换
        for col in ['Price', 'Day BLUE', 'Week BLUE', 'Month BLUE', 'Stop Loss', 'ADX', 'Turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 从 stock_info 缓存补充缺失的名称和行业
        symbols_need_info = df[df['Name'].isna() | (df['Name'] == '')]['Ticker'].tolist()
        if symbols_need_info:
            stock_info_cache = get_stock_info_batch(symbols_need_info)
            for idx, row in df.iterrows():
                ticker = row['Ticker']
                if ticker in stock_info_cache and (pd.isna(row.get('Name')) or row.get('Name') == ''):
                    info = stock_info_cache[ticker]
                    df.at[idx, 'Name'] = info.get('name', '')
                    if pd.isna(row.get('Industry')) or row.get('Industry') == '':
                        df.at[idx, 'Industry'] = info.get('industry', '')
        
        return df, scan_date
    except Exception as e:
        import traceback
        st.error(f"数据库读取失败: {e}")
        st.code(traceback.format_exc())
        return None, None


def load_latest_scan_results():
    """加载最新的扫描结果 - 优先从数据库，回退到 CSV"""
    # 首先尝试从数据库加载
    try:
        init_db()  # 确保数据库已初始化
        stats = get_db_stats()
        if stats and stats['total_records'] > 0:
            # 数据库有数据，使用数据库
            return load_scan_results_from_db()
    except:
        pass
    
    # 回退到 CSV 文件
    files = glob.glob(os.path.join(current_dir, "enhanced_scan_results_*.csv"))
    if not files:
        return None, None
    
    latest_file = max(files, key=os.path.getsize)
    
    try:
        df = pd.read_csv(latest_file)
        
        col_map = {
            'Symbol': 'Ticker',
            'Blue_Daily': 'Day BLUE',
            'Blue_Weekly': 'Week BLUE',
            'Blue_Monthly': 'Month BLUE',
            'Stop_Loss': 'Stop Loss',
            'Shares_Rec': 'Shares Rec',
            'VP_Rating': 'Vol Profile',
            'Market_Cap': 'Mkt Cap Raw',
            'Company_Name': 'Name',
            'Industry': 'Industry',
            'Turnover_M': 'Turnover'
        }
        df.rename(columns=col_map, inplace=True)
        
        if 'Mkt Cap Raw' in df.columns:
            df['Mkt Cap'] = pd.to_numeric(df['Mkt Cap Raw'], errors='coerce').fillna(0) / 1_000_000_000
        else:
            df['Mkt Cap'] = 0.0
            
        def get_strategy_label(row):
            strategies = []
            if row.get('Strat_D_Trend', False):
                strategies.append('Trend-D')
            if row.get('Strat_C_Resonance', False):
                strategies.append('Resonance-C')
            if not strategies and row.get('Legacy_Signal', False):
                strategies.append('Legacy')
            return " | ".join(strategies) if strategies else "N/A"
            
        df['Strategy'] = df.apply(get_strategy_label, axis=1)
        
        def calculate_score(row):
            score = 0
            blue = row.get('Day BLUE', 0)
            score += min(blue / 200, 1.0) * 40
            adx = row.get('ADX', 0)
            score += min(adx / 60, 1.0) * 30
            pr = row.get('Profit_Ratio', 0.5)
            score += pr * 30
            return int(score)
            
        if 'Score' not in df.columns:
            df['Score'] = df.apply(calculate_score, axis=1)

        for col in ['Price', 'Day BLUE', 'Week BLUE', 'Stop Loss', 'Score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, os.path.basename(latest_file)
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None, None


def render_market_pulse(market='US'):
    """
    Market Pulse Dashboard - 显示大盘指数状态
    US: SPY/QQQ/DIA/IWM + VIX
    CN: 上证/深证/创业板/沪深300
    """
    from data_fetcher import get_cn_index_data
    
    # 缓存键 (每10分钟刷新, 按市场区分)
    from datetime import datetime
    cache_time_key = datetime.now().strftime("%Y%m%d%H") + str(datetime.now().minute // 10)
    cache_key = f"market_pulse_{market}_{cache_time_key}"
    
    # 检查缓存
    if cache_key not in st.session_state:
        # 根据市场选择指数
        if market == 'CN':
            indices = {
                '000001.SH': {'name': '上证指数', 'emoji': '🔴'},
                '399001.SZ': {'name': '深证成指', 'emoji': '🟢'},
                '399006.SZ': {'name': '创业板指', 'emoji': '💡'},
                '000300.SH': {'name': '沪深300', 'emoji': '📊'},
            }
            data_fetcher = get_cn_index_data
            currency = '¥'
        else:
            indices = {
                'SPY': {'name': 'S&P 500', 'emoji': '📊'},
                'QQQ': {'name': 'Nasdaq 100', 'emoji': '💻'},
                'DIA': {'name': 'Dow 30', 'emoji': '🏭'},
                'IWM': {'name': 'Russell 2000', 'emoji': '🏢'},
            }
            data_fetcher = fetch_data_from_polygon
            currency = '$'
        
        index_data = {}
        index_data['_currency'] = currency
        index_data['_market'] = market
        
        for symbol, info in indices.items():
            try:
                # 获取日线数据
                df_daily = data_fetcher(symbol, days=100)
                
                if df_daily is not None and len(df_daily) >= 30:
                    # 计算日线 BLUE
                    blue_daily = calculate_blue_signal_series(
                        df_daily['Open'].values,
                        df_daily['High'].values,
                        df_daily['Low'].values,
                        df_daily['Close'].values
                    )
                    
                    # 计算周线 BLUE
                    df_weekly = df_daily.resample('W-MON').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                    
                    blue_weekly = [0]
                    if len(df_weekly) >= 10:
                        blue_weekly = calculate_blue_signal_series(
                            df_weekly['Open'].values,
                            df_weekly['High'].values,
                            df_weekly['Low'].values,
                            df_weekly['Close'].values
                        )
                    
                    # 计算筹码形态
                    chip_result = quick_chip_analysis(df_daily)
                    chip_pattern = chip_result.get('label', '') if chip_result else ''
                    
                    # 最新价格和变化
                    latest_price = df_daily['Close'].iloc[-1]
                    prev_price = df_daily['Close'].iloc[-2] if len(df_daily) > 1 else latest_price
                    price_change = (latest_price - prev_price) / prev_price * 100
                    
                    index_data[symbol] = {
                        'name': info['name'],
                        'emoji': info['emoji'],
                        'price': latest_price,
                        'change': price_change,
                        'day_blue': blue_daily[-1] if len(blue_daily) > 0 else 0,
                        'week_blue': blue_weekly[-1] if len(blue_weekly) > 0 else 0,
                        'chip': chip_pattern
                    }
            except Exception as e:
                index_data[symbol] = {
                    'name': info['name'],
                    'emoji': info['emoji'],
                    'price': 0,
                    'change': 0,
                    'day_blue': 0,
                    'week_blue': 0,
                    'chip': '',
                    'error': str(e)
                }
        
        # VIX 数据 (仅美股, 使用 VIXY ETF 因为 VIX 直接指数无法获取)
        if market == 'US':
            try:
                vix_df = fetch_data_from_polygon('VIXY', days=30)
                if vix_df is not None and len(vix_df) > 0:
                    vix_price = vix_df['Close'].iloc[-1]
                    vix_prev = vix_df['Close'].iloc[-2] if len(vix_df) > 1 else vix_price
                    vix_change = vix_price - vix_prev
                    
                    # VIXY 的阈值需要调整 (ETF 价格不同于 VIX 指数)
                    if vix_price < 20:
                        vix_mood = "😌 极度贪婪"
                    elif vix_price < 25:
                        vix_mood = "🙂 平静"
                    elif vix_price < 30:
                        vix_mood = "😐 中性"
                    elif vix_price < 40:
                        vix_mood = "😟 焦虑"
                    else:
                        vix_mood = "😱 恐惧"
                        
                    index_data['VIX'] = {
                        'price': vix_price,
                        'change': vix_change,
                        'mood': vix_mood
                    }
                else:
                    index_data['VIX'] = {'price': 0, 'change': 0, 'mood': '数据不可用'}
            except:
                index_data['VIX'] = {'price': 0, 'change': 0, 'mood': '未知'}
        
        # 商品/加密资产数据 (仅美股: Gold, Silver, BTC)
        if market == 'US':
            alt_assets = {
                'GLD': {'name': '黄金', 'emoji': '🥇', 'format': '${:.2f}'},
                'SLV': {'name': '白银', 'emoji': '🥈', 'format': '${:.2f}'},
                'X:BTCUSD': {'name': 'BTC', 'emoji': '₿', 'format': '${:,.0f}'}
            }
            
            for symbol, info in alt_assets.items():
                try:
                    df = fetch_data_from_polygon(symbol, days=30)
                    if df is not None and len(df) > 0:
                        price = df['Close'].iloc[-1]
                        prev_price = df['Close'].iloc[-2] if len(df) > 1 else price
                        change = (price - prev_price) / prev_price * 100
                        
                        index_data[symbol] = {
                            'name': info['name'],
                            'emoji': info['emoji'],
                            'price': price,
                            'change': change,
                            'format': info['format']
                        }
                except:
                    index_data[symbol] = {
                        'name': info['name'],
                        'emoji': info['emoji'],
                        'price': 0,
                        'change': 0,
                        'format': info['format']
                    }
        
        # 计算市场情绪综合评分
        # 过滤掉私有键和VIX，只看主要指数
        main_indices = [k for k in index_data.keys() if not k.startswith('_') and k not in ['VIX', 'GLD', 'SLV', 'X:BTCUSD']]
        bullish_count = sum(1 for k in main_indices if index_data.get(k, {}).get('day_blue', 0) > 100)
        total_indices = len(main_indices)
        
        vix_ok = index_data.get('VIX', {}).get('price', 20) < 25 if market == 'US' else True
        
        if bullish_count >= 3 and vix_ok:
            market_sentiment = ("🟢 强势做多", "进攻型 60-80%", "#3fb950")
        elif bullish_count >= 2:
            market_sentiment = ("🟡 震荡偏多", "平衡型 40-60%", "#d29922")
        elif bullish_count >= 1:
            market_sentiment = ("🟠 分化观望", "防守型 20-40%", "#f85149")
        else:
            market_sentiment = ("🔴 弱势防守", "空仓或对冲", "#f85149")
        
        index_data['_sentiment'] = market_sentiment
        index_data['_bullish_count'] = bullish_count
        
        st.session_state[cache_key] = index_data
    else:
        index_data = st.session_state[cache_key]
    
    # === UI 渲染 ===
    with st.container():
        market = index_data.get('_market', 'US')
        currency = index_data.get('_currency', '$')
        
        market_title = "🇺🇸 US Market Pulse" if market == 'US' else "🇨🇳 A股大盘"
        st.markdown(f"### {market_title}")
        
        # 根据市场动态选择要显示的指数
        if market == 'CN':
            display_symbols = ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH']
            col_count = 4
        else:
            display_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VIX']
            col_count = 5
        
        cols = st.columns(col_count)
        
        for i, (symbol, col) in enumerate(zip(display_symbols, cols)):
            with col:
                data = index_data.get(symbol, {})
                
                if symbol == 'VIX':
                    # VIX 特殊显示
                    price = data.get('price', 0)
                    change = data.get('change', 0)
                    mood = data.get('mood', '')
                    
                    delta_color = "inverse" if change < 0 else "normal"
                    st.metric(
                        label="VIX 恐惧指数",
                        value=f"{price:.1f}",
                        delta=f"{change:+.1f}",
                        delta_color=delta_color
                    )
                    st.caption(mood)
                else:
                    # 常规指数
                    price = data.get('price', 0)
                    change = data.get('change', 0)
                    day_blue = data.get('day_blue', 0)
                    week_blue = data.get('week_blue', 0)
                    chip = data.get('chip', '')
                    name = data.get('name', symbol)
                    emoji = data.get('emoji', '')
                    
                    # 趋势图标
                    if change > 0.5:
                        trend = "📈"
                    elif change < -0.5:
                        trend = "📉"
                    else:
                        trend = "➡️"
                    
                    # 显示标签：A股显示名称，美股显示代码
                    if market == 'CN':
                        display_label = f"{emoji} {name} {trend}"
                    else:
                        display_label = f"{symbol} {trend}"
                    
                    st.metric(
                        label=display_label,
                        value=f"{currency}{price:.2f}",
                        delta=f"{change:+.2f}%"
                    )
                    
                    # BLUE 信号 + 筹码
                    blue_text = f"D:{day_blue:.0f} W:{week_blue:.0f}"
                    if chip:
                        blue_text += f" {chip}"
                    
                    # 颜色编码
                    if day_blue > 100:
                        st.markdown(f"<span style='color:#3fb950;font-size:0.85rem;'>{blue_text}</span>", unsafe_allow_html=True)
                    elif day_blue > 50:
                        st.markdown(f"<span style='color:#d29922;font-size:0.85rem;'>{blue_text}</span>", unsafe_allow_html=True)
                    else:
                        st.caption(blue_text)
        
        # === 第二行: 商品/加密资产 (仅美股) ===
        if market == 'US':
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            alt_cols = st.columns(4)
            
            for i, (symbol, col) in enumerate(zip(['GLD', 'SLV', 'X:BTCUSD'], alt_cols[:3])):
                with col:
                    data = index_data.get(symbol, {})
                    price = data.get('price', 0)
                    change = data.get('change', 0)
                    name = data.get('name', symbol)
                    emoji = data.get('emoji', '')
                    fmt = data.get('format', '${:.2f}')
                    
                    # 趋势图标
                    if change > 0.5:
                        trend = "📈"
                    elif change < -0.5:
                        trend = "📉"
                    else:
                        trend = "➡️"
                    
                    # 格式化价格
                    try:
                        formatted_price = fmt.format(price)
                    except:
                        formatted_price = f"${price:.2f}"
                    
                    st.metric(
                        label=f"{emoji} {name} {trend}",
                        value=formatted_price,
                        delta=f"{change:+.2f}%"
                    )
        
        # === 北向资金 (仅 A股) ===
        if market == 'CN':
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            
            # 尝试从缓存获取北向资金数据
            north_cache_key = f"north_money_{cache_time_key}"
            
            if north_cache_key not in st.session_state:
                try:
                    from data_fetcher import get_north_money_today
                    north_data = get_north_money_today()
                    st.session_state[north_cache_key] = north_data
                except Exception as e:
                    st.session_state[north_cache_key] = {}
            else:
                north_data = st.session_state[north_cache_key]
            
            if north_data:
                north_cols = st.columns(5)
                north_source = north_data.get("source", "")
                north_note = north_data.get("note", "")
                north_fallback = bool(north_data.get("is_fallback", False))
                north_suspect = bool(north_data.get("is_suspect_zero", False))
                official_val = north_data.get("official_north_money")
                official_date = north_data.get("official_date", "--")
                
                with north_cols[0]:
                    north_val = north_data.get('north_money', 0)
                    if north_val > 0:
                        icon = "📈"
                        delta_text = "净流入"
                        delta_color = "normal"
                    elif north_val < 0:
                        icon = "📉"
                        delta_text = "净流出"
                        delta_color = "inverse"
                    else:
                        icon = "➖"
                        delta_text = "持平/待更新"
                        delta_color = "off"
                    st.metric(
                        label=f"🏦 北向资金 {icon}",
                        value=f"¥{abs(north_val):.2f}亿",
                        delta=delta_text,
                        delta_color=delta_color
                    )
                
                with north_cols[1]:
                    sh_val = north_data.get('sh_money', 0)
                    if pd.isna(sh_val):
                        sh_val = 0
                        sh_delta = "N/A"
                        sh_color = "off"
                    else:
                        if sh_val > 0:
                            sh_delta = "流入"
                            sh_color = "normal"
                        elif sh_val < 0:
                            sh_delta = "流出"
                            sh_color = "inverse"
                        else:
                            sh_delta = "持平"
                            sh_color = "off"
                    st.metric(
                        label="沪股通",
                        value=f"¥{abs(sh_val):.2f}亿",
                        delta=sh_delta,
                        delta_color=sh_color
                    )
                
                with north_cols[2]:
                    sz_val = north_data.get('sz_money', 0)
                    if pd.isna(sz_val):
                        sz_val = 0
                        sz_delta = "N/A"
                        sz_color = "off"
                    else:
                        if sz_val > 0:
                            sz_delta = "流入"
                            sz_color = "normal"
                        elif sz_val < 0:
                            sz_delta = "流出"
                            sz_color = "inverse"
                        else:
                            sz_delta = "持平"
                            sz_color = "off"
                    st.metric(
                        label="深股通",
                        value=f"¥{abs(sz_val):.2f}亿",
                        delta=sz_delta,
                        delta_color=sz_color
                    )
                
                with north_cols[3]:
                    # 权威(日终)值
                    if official_val is None or (isinstance(official_val, float) and pd.isna(official_val)):
                        st.metric("权威(日终)", "N/A", "待更新", delta_color="off")
                    else:
                        off_val = float(official_val)
                        if off_val > 0:
                            off_delta = "净流入"
                            off_color = "normal"
                        elif off_val < 0:
                            off_delta = "净流出"
                            off_color = "inverse"
                        else:
                            off_delta = "持平"
                            off_color = "off"
                        st.metric(
                            label="权威(日终)",
                            value=f"¥{abs(off_val):.2f}亿",
                            delta=off_delta,
                            delta_color=off_color,
                        )
                
                with north_cols[4]:
                    st.caption(f"📅 {north_data.get('date', '--')}")
                    st.caption(f"权威日: {official_date}")
                    if north_val == 0:
                        st.caption("ℹ️ 可能因港股通未开盘/当日未更新")
                    if north_source:
                        st.caption(f"源: {north_source}")
                    if north_note:
                        st.caption(f"注: {north_note}")
                    # 北向资金判断
                    if north_val > 50:
                        st.markdown("🟢 **大幅流入**")
                    elif north_val > 0:
                        st.markdown("🟡 **小幅流入**")
                    elif north_val == 0:
                        st.markdown("⚪ **中性**")
                    elif north_val > -50:
                        st.markdown("🟠 **小幅流出**")
                    else:
                        st.markdown("🔴 **大幅流出**")

                if north_suspect and not north_fallback:
                    st.warning("⚠️ 北向实时数据疑似异常归零，当前仅供参考。")
                elif north_fallback:
                    st.info("ℹ️ 北向资金已使用回退值（最近有效数据）。")
        
        # 市场情绪总结
        sentiment = index_data.get('_sentiment', ('未知', '未知', 'gray'))
        bullish = index_data.get('_bullish_count', 0)
        
        st.markdown(f"""
        <div style="background: rgba(22, 27, 34, 0.8); border-radius: 8px; padding: 12px 16px; margin-top: 10px; border-left: 4px solid {sentiment[2]};">
            <span style="font-size: 1.1rem; font-weight: 600;">{sentiment[0]}</span>
            <span style="color: #8b949e; margin-left: 12px;">建议仓位: {sentiment[1]}</span>
            <span style="color: #8b949e; margin-left: 12px;">({bullish}/4 指数有日BLUE信号)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # === 指数详情展开 ===
        with st.expander("🔍 查看指数/资产详情 (筹码分布 & 资金流向)", expanded=False):
            # 可选指数列表
            all_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'SLV', 'X:BTCUSD']
            symbol_labels = {
                'SPY': '📊 SPY (S&P 500)',
                'QQQ': '💻 QQQ (Nasdaq 100)',
                'DIA': '🏭 DIA (Dow 30)',
                'IWM': '🏢 IWM (Russell 2000)',
                'GLD': '🥇 GLD (黄金)',
                'SLV': '🥈 SLV (白银)',
                'X:BTCUSD': '₿ BTC (比特币)'
            }
            
            selected_index = st.selectbox(
                "选择要分析的指数/资产",
                options=all_symbols,
                format_func=lambda x: symbol_labels.get(x, x),
                key="market_pulse_index_detail"
            )
            
            if selected_index:
                with st.spinner(f"正在加载 {selected_index} 数据..."):
                    try:
                        # 获取数据
                        df_detail = fetch_data_from_polygon(selected_index, days=120)
                        
                        if df_detail is not None and len(df_detail) >= 30:
                            detail_cols = st.columns([2, 1])
                            
                            with detail_cols[0]:
                                # K线图 + BLUE 信号
                                st.markdown("##### 📈 K线图 & BLUE信号")
                                fig = create_candlestick_chart_dynamic(
                                    df_full=df_detail,
                                    df_for_vp=df_detail,
                                    symbol=selected_index,
                                    name=symbol_labels.get(selected_index, selected_index),
                                    period='daily',
                                    show_volume_profile=True
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("无法生成图表")
                            
                            with detail_cols[1]:
                                # 筹码分析摘要
                                st.markdown("##### 📊 筹码分析")
                                chip_result = quick_chip_analysis(df_detail)
                                
                                if chip_result:
                                    poc_pos = chip_result.get('poc_position', 50)
                                    bottom_ratio = chip_result.get('bottom_chip_ratio', 0) * 100
                                    max_chip = chip_result.get('max_chip_pct', 0)
                                    is_strong = chip_result.get('is_strong_bottom_peak', False)
                                    is_peak = chip_result.get('is_bottom_peak', False)
                                    
                                    # 显示指标
                                    st.metric("POC 位置", f"{poc_pos:.1f}%", help="成本峰值在价格区间的位置")
                                    st.metric("底部筹码", f"{bottom_ratio:.1f}%", help="底部30%价格区间的筹码占比")
                                    st.metric("单峰最大", f"{max_chip:.1f}%", help="最大筹码柱占比")
                                    
                                    if is_strong:
                                        st.success("🔥 强势顶格峰")
                                    elif is_peak:
                                        st.info("📍 底部密集")
                                    else:
                                        st.caption("普通形态")
                                else:
                                    st.warning("无法计算筹码分布")
                            
                            # 资金流向图表 (使用筹码流动对比)
                            st.markdown("##### 💰 筹码流动对比 (30天前 vs 现在)")
                            chip_flow_data = analyze_chip_flow(df_detail, lookback_days=30)
                            if chip_flow_data:
                                flow_fig = create_chip_flow_chart(chip_flow_data, selected_index)
                                if flow_fig:
                                    st.plotly_chart(flow_fig, use_container_width=True)
                            else:
                                st.info("数据不足，无法显示筹码流动")
                                
                        else:
                            st.warning(f"无法获取 {selected_index} 数据")
                            
                    except Exception as e:
                        st.error(f"加载失败: {e}")
        
        st.divider()


def render_market_news_intel(market='US', unique_key=''):
    """📰 大盘新闻智能面板 — 嵌入工作台/扫描页顶部
    
    按需加载(expander)，缓存10分钟，不阻塞主页面。
    """
    from datetime import datetime
    
    cache_time_key = datetime.now().strftime("%Y%m%d%H") + str(datetime.now().minute // 10)
    cache_key = f"market_news_intel_{market}_{cache_time_key}"
    
    with st.expander("📰 大盘新闻 & AI 分析", expanded=False):
        if cache_key in st.session_state:
            _display_market_news_cached(st.session_state[cache_key], market)
            return
        
        if st.button("🔄 加载大盘新闻", key=f"load_market_news_{unique_key}_{market}",
                     use_container_width=True, type="primary"):
            with st.spinner("正在获取大盘新闻和 AI 分析..."):
                result = _fetch_market_news_intel(market)
                st.session_state[cache_key] = result
                _display_market_news_cached(result, market)
        else:
            st.caption("💡 点击上方按钮加载大盘新闻和 AI 分析")


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_market_news_intel(market: str) -> dict:
    """抓取大盘指数的新闻 + 生成 AI 摘要"""
    try:
        from news.crawler import get_news_crawler
        from news.classifier import get_event_classifier
        from news.models import Sentiment
    except ImportError:
        return {'error': '新闻模块不可用'}
    
    # 选择大盘标的
    if market == 'CN':
        symbols = [('000001.SH', '上证指数'), ('000300.SH', '沪深300')]
        search_queries = [('A股 大盘 今日', ''), ('沪深300 行情', '')]
    else:
        symbols = [('SPY', 'S&P 500'), ('QQQ', 'Nasdaq')]
        search_queries = None
    
    crawler = get_news_crawler()
    classifier = get_event_classifier(use_llm=True)
    
    all_events = []
    for symbol, name in symbols:
        try:
            events = crawler.crawl_all(symbol, company_name=name,
                                       market=market, max_per_source=3)
            all_events.extend(events)
        except Exception as e:
            pass
    
    # 去重 + 分类
    seen = set()
    unique_events = []
    for e in all_events:
        key = e.title[:30].lower()
        if key not in seen:
            seen.add(key)
            unique_events.append(e)
    
    if unique_events:
        unique_events = classifier.classify_batch(unique_events)
    
    # 统计情绪
    bull = sum(1 for e in unique_events if e.sentiment.score > 0)
    bear = sum(1 for e in unique_events if e.sentiment.score < 0)
    neutral = sum(1 for e in unique_events if e.sentiment.score == 0)
    
    total = bull + bear
    ratio = (bull - bear) / total if total > 0 else 0
    
    # 社交热度 (仅美股)
    social_buzz = {}
    if market == 'US':
        try:
            from news.crawler import ApeWisdomCrawler
            ape = ApeWisdomCrawler()
            wsb_top = ape.get_trending(limit=5)
            social_buzz['wsb_top'] = wsb_top
        except:
            pass
    
    # AI 总结 (如有 Gemini)
    ai_summary = ""
    try:
        from ml.llm_intelligence import get_gemini_model
        model = get_gemini_model()
        if model and unique_events:
            headlines = "\n".join([
                f"- [{e.sentiment.emoji}] {e.title[:60]}"
                for e in unique_events[:8]
            ])
            prompt = f"""根据以下{market}市场最新新闻，用3-4句话总结今日市场情绪和关键驱动因素。
直接给出结论，不要废话。

{headlines}

格式: 情绪判断 + 关键事件 + 操作建议"""
            
            resp = model.generate_content(prompt)
            ai_summary = resp.text[:300] if resp.text else ""
    except Exception as e:
        ai_summary = ""
    
    return {
        'events': [e.to_dict() for e in unique_events[:10]],
        'bull': bull,
        'bear': bear,
        'neutral': neutral,
        'ratio': ratio,
        'ai_summary': ai_summary,
        'social_buzz': social_buzz,
    }


def _display_market_news_cached(result: dict, market: str):
    """渲染缓存的大盘新闻结果"""
    if result.get('error'):
        st.warning(result['error'])
        return
    
    bull = result.get('bull', 0)
    bear = result.get('bear', 0)
    neutral = result.get('neutral', 0)
    ratio = result.get('ratio', 0)
    ai_summary = result.get('ai_summary', '')
    events = result.get('events', [])
    social_buzz = result.get('social_buzz', {})
    
    # 情绪指示器
    if ratio > 0.3:
        color, emoji, label = "#00C853", "🟢", "偏多"
    elif ratio < -0.3:
        color, emoji, label = "#FF1744", "🔴", "偏空"
    else:
        color, emoji, label = "#FFD600", "⚪", "中性"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
**{emoji} 大盘新闻情绪: {label}** — 📈{bull} 利好 · 📉{bear} 利空 · ➖{neutral} 中性
        """)
        
        if ai_summary:
            st.info(f"🤖 **AI 摘要:** {ai_summary}")
    
    with col2:
        st.metric("情绪比值", f"{ratio:+.2f}")
    
    # Reddit WSB 热门 (美股)
    wsb_top = social_buzz.get('wsb_top', [])
    if wsb_top:
        wsb_str = " · ".join([
            f"**{s['symbol']}**({s.get('mentions',0)})" for s in wsb_top[:5]
        ])
        st.caption(f"🦍 Reddit 热门: {wsb_str}")
    
    # 新闻列表(折叠)
    if events:
        for ev in events[:5]:
            sent_emoji = {"very_bullish": "🔥", "bullish": "📈", "neutral": "➖",
                         "bearish": "📉", "very_bearish": "💥"}.get(ev.get('sentiment', ''), '➖')
            title = ev.get('title', '')[:60]
            source = ev.get('source', '')[:20]
            st.caption(f"{sent_emoji} {title} — *{source}*")



def get_market_mood(df):
    """根据扫描结果判断市场情绪"""
    if df is None or df.empty:
        return "未知", "gray"
    
    high_score_count = len(df[df['Score'] >= 85])
    total_count = len(df)
    ratio = high_score_count / total_count if total_count > 0 else 0
    
    if total_count > 50: 
        if ratio > 0.3:
            return "🔥 极度火热 (FOMO)", "red"
        elif ratio > 0.15:
            return "☀️ 积极做多", "orange"
        elif ratio > 0.05:
            return "☁️ 震荡分化", "blue"
        else:
            return "❄️ 冰点/观望", "lightblue"
    else:
        return f"扫描样本数: {total_count}", "gray"

# --- 缓存: 逃顶&买入预警扫描 (避免每次交互重算 30 只股票) ---

@st.cache_data(ttl=900, show_spinner=False)
def _cached_escape_warning_scan(symbols: tuple, market: str):
    """缓存逃顶/买入预警扫描结果 (15分钟TTL)。

    对每只股票拉取 250 天数据并计算 phantom/heima_full/ADX 指标，
    筛选三重逃顶、PINK逃顶、黄金底、趋势回调买入信号。
    symbols 使用 tuple 以满足 st.cache_data 的 hashable 要求。
    """
    from indicator_utils import calculate_phantom_indicator, calculate_adx_series, calculate_heima_full
    from data_fetcher import get_stock_data

    escape_warnings = []
    trend_pullbacks = []
    golden_bottoms = []

    for sym in symbols:
        try:
            sym_data = get_stock_data(sym, market=market, days=250)
            if sym_data is None or len(sym_data) < 50:
                continue
            ph = calculate_phantom_indicator(
                sym_data['Open'].values, sym_data['High'].values,
                sym_data['Low'].values, sym_data['Close'].values,
                sym_data['Volume'].values
            )
            hf = calculate_heima_full(
                sym_data['High'].values, sym_data['Low'].values,
                sym_data['Close'].values, sym_data['Open'].values
            )
            adx_arr = calculate_adx_series(
                sym_data['High'].values, sym_data['Low'].values,
                sym_data['Close'].values
            )
            adx_v = float(adx_arr[-1])
            pink_v = float(ph['pink'][-1])
            is_sell = bool(ph['sell_signal'][-1])
            green_v = float(ph['green'][-1])
            is_blue_dis = bool(ph['blue_disappear'][-1])

            price_sym = "¥" if market == "CN" else "$"
            price_val = float(sym_data['Close'].iloc[-1])
            display_name = sym.split('.')[0] if '.' in sym else sym

            # 三重逃顶 (最强)
            has_top_div = bool(hf['top_divergence'][-1])
            if has_top_div and pink_v > 80 and green_v < 0:
                escape_warnings.append({
                    'symbol': sym, 'name': display_name,
                    'pink': pink_v, 'price': price_val, 'price_sym': price_sym,
                    'level': 'critical', 'reason': '三重逃顶 (86%)'
                })
            elif is_sell and green_v < 0 and adx_v < 30:
                escape_warnings.append({
                    'symbol': sym, 'name': display_name,
                    'pink': pink_v, 'price': price_val, 'price_sym': price_sym,
                    'level': 'high', 'reason': 'PINK逃顶+流出'
                })
            elif has_top_div:
                escape_warnings.append({
                    'symbol': sym, 'name': display_name,
                    'pink': pink_v, 'price': price_val, 'price_sym': price_sym,
                    'level': 'low', 'reason': '顶背离(需确认)'
                })

            # 黄金底
            has_golden = bool(hf['golden_bottom'][-1])
            cci_val = float(hf['CCI'][-1])
            if has_golden:
                golden_bottoms.append({
                    'symbol': sym, 'name': display_name,
                    'cci': cci_val, 'price': price_val, 'price_sym': price_sym,
                })

            # 趋势回调买入
            if is_blue_dis and adx_v >= 25:
                trend_pullbacks.append({
                    'symbol': sym, 'name': display_name,
                    'adx': adx_v, 'price': price_val, 'price_sym': price_sym,
                })
        except Exception:
            continue

    return escape_warnings, golden_bottoms, trend_pullbacks


# --- 页面逻辑 ---

def render_todays_picks_page():
    """🎯 每日工作台 - SOP 三步法驱动的交易工作流"""
    st.header("🎯 每日工作台")
    st.caption("① 处理风险 → ② 执行买入 → ③ 收盘复盘")
    
    try:
        _render_todays_picks_page_inner()
    except Exception as e:
        import traceback
        st.error(f"⚠️ 每日工作台加载异常: {e}")
        st.code(traceback.format_exc(), language='text')
        st.info("请刷新页面重试。若持续出错请截图以上错误信息反馈。")


def _render_todays_picks_page_inner():
    """每日工作台完整渲染逻辑 (内部函数)"""
    
    # 导入模块
    try:
        from strategies.decision_system import get_strategy_manager
        from strategies.performance_tracker import get_all_strategy_performance
    except ImportError as e:
        st.warning(f"策略模块导入失败（已降级）: {e}")

        class _DummyStrategyManager:
            def get_daily_picks(self, *args, **kwargs):
                return {}

        def get_strategy_manager():
            return _DummyStrategyManager()

        def get_all_strategy_performance():
            return []
    
    from db.database import (
        query_scan_results,
        get_scanned_dates,
        get_stock_info_batch,
    )
    try:
        from db.database import get_app_setting, set_app_setting
    except Exception:
        # 兼容云端缓存旧版本 database.py 的场景
        def get_app_setting(_key, default=None):
            return default

        def set_app_setting(_key, _value):
            return None
    from services.portfolio_service import get_portfolio_summary
    try:
        from services.candidate_tracking_service import (
            capture_daily_candidates,
            refresh_candidate_tracking,
            get_candidate_tracking_rows,
            build_combo_stats,
            build_segment_stats,
            reclassify_tracking_tags,
            derive_signal_tags,
            evaluate_exit_rule,
            clear_all_caches as clear_tracking_caches,
            CORE_TAGS,
            DEFAULT_TAG_RULES,
            backfill_candidates_from_scan_history,
        )
        from services.meta_allocator_service import (
            evaluate_strategy_baskets,
            evaluate_strategy_baskets_best_exit,
            allocate_meta_weights,
            build_today_meta_plan,
        )
    except Exception as e:
        st.warning(f"候选追踪模块加载失败（已降级）: {e}")

        def capture_daily_candidates(*args, **kwargs):
            return 0

        def refresh_candidate_tracking(*args, **kwargs):
            return 0

        def get_candidate_tracking_rows(*args, **kwargs):
            return []

        def build_combo_stats(*args, **kwargs):
            return []

        def build_segment_stats(*args, **kwargs):
            return []

        def reclassify_tracking_tags(*args, **kwargs):
            return 0
        
        def derive_signal_tags(*args, **kwargs):
            return []
        
        def evaluate_exit_rule(*args, **kwargs):
            return {
                "sample": 0,
                "win_rate_pct": None,
                "avg_return_pct": None,
                "avg_exit_day": None,
                "avg_first_profit_day": None,
                "avg_first_nonprofit_day": None,
                "avg_profit_span_days": None,
            }

        def backfill_candidates_from_scan_history(*args, **kwargs):
            return 0

        def clear_tracking_caches():
            pass

        def evaluate_strategy_baskets(*args, **kwargs):
            return []

        def evaluate_strategy_baskets_best_exit(*args, **kwargs):
            return []

        def allocate_meta_weights(*args, **kwargs):
            return []

        def build_today_meta_plan(*args, **kwargs):
            return []

        CORE_TAGS = []
        DEFAULT_TAG_RULES = {
            "day_blue_min": 100.0,
            "week_blue_min": 80.0,
            "month_blue_min": 60.0,
            "chip_dense_profit_ratio_min": 0.7,
            "chip_breakout_profit_ratio_min": 0.9,
            "chip_overhang_profit_ratio_max": 0.3,
        }
    
    # 尝试导入工作流服务
    try:
        from services.daily_workflow import (
            get_workflow_service, get_today_tasks, 
            get_signal_pipeline, get_daily_summary
        )
        workflow_available = True
    except ImportError:
        workflow_available = False
    
    # 侧边栏: 设置
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ 工作台设置")
        
        market_choice = st.radio("市场", ["🇺🇸 美股", "🇨🇳 A股"], horizontal=True, key="picks_market")
        market = "US" if "美股" in market_choice else "CN"
        _set_active_market(market)
        
        # 检测市场切换，清除之前选中的股票
        prev_market = st.session_state.get('_picks_prev_market', market)
        if prev_market != market:
            # 清除所有选中状态
            for key in ['action_selected_symbol', 'action_buy_symbol', 'discover_selected', 
                       'portfolio_selected', 'portfolio_sell', 'portfolio_add']:
                if key in st.session_state:
                    st.session_state[key] = None
            st.session_state['_picks_prev_market'] = market
        else:
            st.session_state['_picks_prev_market'] = market
        
        top_n = st.slider("每策略选股数", 3, 20, 8, key="picks_topn")
        
        show_performance = st.checkbox("显示策略历史表现", value=True, key="picks_perf")
        show_backtest = st.checkbox("显示回测追踪", value=False, key="picks_backtest")
    
    # ============================================
    # 📊 大盘环境 (Market Pulse + News)
    # ============================================
    render_market_pulse(market=market)
    render_market_news_intel(market=market, unique_key='picks')
    
    # ============================================
    # 📊 顶部: 行动摘要卡片
    # ============================================
    # --- 轻量日期计数：1次 GROUP BY 代替 30次全量查询 ---
    try:
        date_count_rows = _cached_scan_date_counts(market=market, limit=30)
    except Exception as e:
        st.warning(f"读取扫描日期失败，已降级为空数据模式: {e}")
        date_count_rows = []

    if not date_count_rows:
        # 回退：尝试 get_scanned_dates
        try:
            fall_dates = _cached_scanned_dates(market=market) or []
        except Exception:
            fall_dates = []
        if fall_dates:
            date_count_rows = [{'scan_date': d, 'count': 0} for d in fall_dates[:30]]
        else:
            st.warning(f"暂无 {market} 市场扫描数据，已进入空数据模式（页面结构保留）。")
            date_count_rows = [{'scan_date': datetime.now().strftime('%Y-%m-%d'), 'count': 0}]

    recent_dates = [r['scan_date'] for r in date_count_rows]
    date_rows_map = {r['scan_date']: r['count'] for r in date_count_rows}

    # 找到第一个有足够数据(>=30)的日期作为默认选择
    preferred_date = recent_dates[0]
    for r in date_count_rows:
        if r['count'] >= 30:
            preferred_date = r['scan_date']
            break

    with st.sidebar:
        date_labels = [
            f"{d} ({date_rows_map.get(d, 0)}条)"
            for d in recent_dates
        ]
        default_idx = 0
        for i, d in enumerate(recent_dates):
            if d == preferred_date:
                default_idx = i
                break
        selected_date_label = st.selectbox(
            "机会扫描日期",
            options=date_labels,
            index=default_idx,
            key=f"picks_scan_date_{market}",
        )
    latest_date = selected_date_label.split(" (", 1)[0]
    try:
        results = _cached_scan_results(scan_date=latest_date, market=market, limit=2000)
    except Exception as e:
        st.warning(f"读取扫描结果失败，已降级为空数据模式: {e}")
        results = []
    # 强兜底：若当前日期为空，自动回退到最近有数据的日期
    if not results:
        # 利用已有的 date_count_rows 找到首个 count>0 的日期，只做1次查询
        fallback_date = None
        for r in date_count_rows:
            if r['count'] > 0 and r['scan_date'] != latest_date:
                fallback_date = r['scan_date']
                break
        if fallback_date:
            try:
                results = _cached_scan_results(scan_date=fallback_date, market=market, limit=2000) or []
            except Exception:
                results = []
            if results:
                latest_date = fallback_date
                st.warning(f"当前选择日期无数据，已自动回退到最近有数据日期：{latest_date}")
        if not results:
            st.error(f"{market} 最近30个扫描日均无可用结果。可先运行 Daily Scan 或检查数据源。")
            results = []
    df = pd.DataFrame(results) if results else pd.DataFrame()
    
    # 补充空头信号列（Supabase 可能还没有 lired_daily/pink_daily）
    if not df.empty and 'pink_daily' not in df.columns:
        try:
            from db.database import get_db
            with get_db() as conn:
                cursor = conn.cursor()
                # 获取所有有 pink/lired 数据的记录
                cursor.execute(
                    "SELECT * FROM scan_results WHERE scan_date=? AND market=? AND (pink_daily IS NOT NULL OR lired_daily IS NOT NULL)",
                    (latest_date, market)
                )
                bearish_rows = cursor.fetchall()
                bearish_data = {row['symbol']: dict(row) for row in bearish_rows}
            
            if bearish_data:
                # 1) 更新现有行的空头信号列
                df['lired_daily'] = df['symbol'].map(lambda s: bearish_data.get(s, {}).get('lired_daily') or 0)
                df['pink_daily'] = df['symbol'].map(lambda s: bearish_data.get(s, {}).get('pink_daily') or 0)
                if 'duokongwang_sell' not in df.columns:
                    df['duokongwang_sell'] = df['symbol'].map(lambda s: bearish_data.get(s, {}).get('duokongwang_sell') or 0)
                
                # 2) 追加 SQLite 里有空头信号但不在当前 df 中的股票
                existing_syms = set(df['symbol'].tolist())
                extra_bearish = [v for k, v in bearish_data.items() 
                                if k not in existing_syms and ((v.get('pink_daily') or 0) > 80 or (v.get('lired_daily') or 0) > 0)]
                if extra_bearish:
                    extra_df = pd.DataFrame(extra_bearish)
                    df = pd.concat([df, extra_df], ignore_index=True)
        except Exception:
            pass
    
    # 获取持仓数据
    try:
        portfolio = get_portfolio_summary() or {}
        positions = portfolio.get('details', [])  # 从 summary 中获取持仓详情
    except:
        positions = []
        portfolio = {}
    
    # 自动收录候选追踪快照（按 symbol+market+date 去重）
    if not df.empty:
        try:
            capture_daily_candidates(
                rows=df.to_dict("records"),
                market=market,
                signal_date=latest_date,
                source="daily_workbench",
            )
        except Exception as e:
            st.caption(f"候选追踪初始化失败: {e}")

    # 计算行动项
    buy_opportunities = 0
    sell_signals = 0
    risk_alerts = 0
    
    # === 多头买入信号 ===
    if not df.empty:
        # 强买入信号: 日BLUE > 100 且 周BLUE > 50
        strong_buy = df[
            (df.get('blue_daily', pd.Series([0]*len(df))) > 100) & 
            (df.get('blue_weekly', pd.Series([0]*len(df))) > 50)
        ]
        buy_opportunities = len(strong_buy)
    
    # === 空头卖出信号（来自扫描数据，不依赖持仓）===
    bearish_alerts = []  # 记录空头预警详情
    if not df.empty:
        for _, row in df.iterrows():
            sym = row.get('symbol', '')
            lired_val = float(row.get('lired_daily', 0) or 0)
            pink_val = float(row.get('pink_daily', 0) or 0)
            dkw_sell = bool(row.get('duokongwang_sell', False))
            
            alerts = []
            if lired_val > 0:
                alerts.append(f"🩷 LIRED逃顶信号({lired_val:.1f})")
            if pink_val > 90:
                alerts.append(f"🩷 PINK>90超买({pink_val:.0f})")
            if dkw_sell:
                alerts.append("🔴 多空王卖出")
            
            if alerts:
                bearish_alerts.append({
                    'symbol': sym,
                    'price': float(row.get('price', 0) or 0),
                    'alerts': alerts,
                    'pink_val': pink_val,
                    'lired_val': lired_val,
                    'blue_daily': float(row.get('blue_daily', 0) or 0),
                    'blue_weekly': float(row.get('blue_weekly', 0) or 0),
                    'adx': float(row.get('adx', 0) or 0),
                    'volatility': float(row.get('volatility', 0) or 0),
                    'wave_phase': str(row.get('wave_phase', '') or ''),
                    'cap_category': str(row.get('cap_category', '') or ''),
                    'company_name': str(row.get('company_name', '') or ''),
                    'urgency': 'high' if (lired_val > 0 and pink_val > 90) or (pink_val >= 97) else 'medium'
                })
        
        sell_signals = len(bearish_alerts)
    
    # === 持仓风险检测 ===
    position_alerts = []
    for pos in positions:
        symbol = pos.get('symbol', '')
        avg_cost = pos.get('avg_cost', 0)
        current_price = pos.get('current_price', 0)
        stop_loss = pos.get('stop_loss', avg_cost * 0.92)  # 默认8%止损
        
        if current_price > 0:
            pnl_pct = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
            
            # 检查止损
            if current_price < stop_loss:
                position_alerts.append({
                    'symbol': symbol,
                    'type': 'stop_loss',
                    'message': f'触及止损 ${current_price:.2f} < ${stop_loss:.2f}',
                    'action': '建议卖出',
                    'urgency': 'high'
                })
                risk_alerts += 1
            
            # 检查大幅亏损
            elif pnl_pct < -10:
                position_alerts.append({
                    'symbol': symbol,
                    'type': 'loss',
                    'message': f'亏损 {pnl_pct:.1f}%',
                    'action': '检查止损',
                    'urgency': 'medium'
                })
                risk_alerts += 1
            
            # 检查是否有卖出信号 (BLUE 转弱)
            if not df.empty and symbol in df['symbol'].values:
                stock_data = df[df['symbol'] == symbol].iloc[0]
                day_blue = stock_data.get('blue_daily', 100)
                if day_blue < 30 and pnl_pct > 5:
                    position_alerts.append({
                        'symbol': symbol,
                        'type': 'signal_weak',
                        'message': f'BLUE信号转弱 ({day_blue:.0f}), 盈利 {pnl_pct:.1f}%',
                        'action': '考虑获利了结',
                        'urgency': 'low'
                    })

    # 历史样本：用于给"今日行动"补充可执行置信度（0 = 不限日期，取全量）
    action_days_back = 0
    action_window_label = "全量历史" if int(action_days_back) <= 0 else f"近{int(action_days_back)}天"
    try:
        tracking_rows_for_action = get_candidate_tracking_rows(market=market, days_back=action_days_back)
    except Exception:
        tracking_rows_for_action = []

    # 自动轻量补样本已禁用（会阻塞页面加载数分钟）；
    # 用户可通过下方手动按钮执行回填

    # 手动重建：避免页面加载阶段触发大规模回填导致阻塞
    latest_scan_sample = len(results) if isinstance(results, list) else 0
    sparse_floor = max(120, latest_scan_sample * 2)
    if len(tracking_rows_for_action) < sparse_floor:
        st.caption(
            f"⚠️ 候选追踪样本偏少（当前 {len(tracking_rows_for_action)}，建议 >= {sparse_floor}）。"
            "如需补齐，点击下方按钮手动执行。"
        )
        if st.button("🔧 手动补齐候选追踪（全量历史）", key=f"manual_rebuild_tracking_{market}"):
            try:
                with st.spinner("回填与刷新中，可能需要1-3分钟..."):
                    added_rows = backfill_candidates_from_scan_history(
                        market=market,
                        recent_days=9999,
                        max_per_day=1200,
                    )
                    refreshed_rows = refresh_candidate_tracking(market=market, max_rows=20000)
                    tracking_rows_for_action = get_candidate_tracking_rows(market=market, days_back=action_days_back)
                st.success(f"已补齐: 回填 {added_rows} 条, 刷新 {refreshed_rows} 条, 当前样本 {len(tracking_rows_for_action)}")
            except Exception as _rebuild_err:
                st.error(f"手动补齐失败: {_rebuild_err}")

    def _calc_signal_reliability(tags: list):
        if not tags or not tracking_rows_for_action:
            return {
                "sample": 0,
                "win_rate": None,
                "avg_pnl": None,
                "grade": "N/A",
                "position_hint": "观察",
            }
        tag_set = set(tags)
        matched = [
            r for r in tracking_rows_for_action
            if tag_set.issubset(set(r.get("signal_tags_list") or []))
        ]
        sample = len(matched)
        if sample == 0:
            return {
                "sample": 0,
                "win_rate": None,
                "avg_pnl": None,
                "grade": "N/A",
                "position_hint": "观察",
            }
        wins = sum(1 for r in matched if float(r.get("pnl_pct") or 0) > 0)
        win_rate = wins / sample * 100.0
        avg_pnl = float(np.mean([float(r.get("pnl_pct") or 0) for r in matched]))
        if sample >= 25 and win_rate >= 62 and avg_pnl > 1.5:
            grade, position_hint = "A", "主仓(40-60%)"
        elif sample >= 15 and win_rate >= 55 and avg_pnl > 0.5:
            grade, position_hint = "B", "半仓(20-40%)"
        elif sample >= 8 and win_rate >= 50:
            grade, position_hint = "C", "试仓(10-20%)"
        else:
            grade, position_hint = "D", "仅观察"
        return {
            "sample": sample,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "grade": grade,
            "position_hint": position_hint,
        }

    # ============================================
    # 🧭 SOP 状态条 — 交易员的每日导航
    # ============================================
    has_risk = (sell_signals > 0 or risk_alerts > 0 or len(position_alerts) > 0)
    total_positions = len(positions)
    total_pnl = portfolio.get('total_pnl_pct', 0)

    # SOP 步骤状态
    step1_icon = "🔴" if has_risk else "✅"
    step1_text = f"处理风险 ({sell_signals + risk_alerts})" if has_risk else "风险已清"
    step2_icon = "⏳" if buy_opportunities > 0 else "○"
    step2_text = f"买入执行 ({buy_opportunities})"
    step3_icon = "○"
    step3_text = "收盘复盘"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 16px 24px; border-radius: 12px; margin-bottom: 16px;
                border: 1px solid #2a2a4a;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 1.3em; font-weight: bold;">
                📅 {latest_date}
            </div>
            <div style="display: flex; gap: 24px; font-size: 0.95em;">
                <span style="{'color: #FF5252; font-weight: bold;' if has_risk else 'color: #00C853;'}">
                    {step1_icon} ① {step1_text}
                </span>
                <span style="color: #FFD600;">
                    {step2_icon} ② {step2_text}
                </span>
                <span style="color: #666;">
                    {step3_icon} ③ {step3_text}
                </span>
            </div>
        </div>
        <div style="display: flex; gap: 32px; margin-top: 12px; color: #888; font-size: 0.85em;">
            <span>🟢 买入机会 <b style="color:#00C853;">{buy_opportunities}</b></span>
            <span>🔴 卖出信号 <b style="color:#FF5252;">{sell_signals}</b></span>
            <span>⚠️ 风险警告 <b style="color:#FF9800;">{risk_alerts}</b></span>
            <span>💼 持仓 <b>{total_positions}</b> {'<span style="color:#00C853;">(' + f'{total_pnl:+.1f}%' + ')</span>' if total_positions > 0 and total_pnl >= 0 else ('<span style="color:#FF5252;">(' + f'{total_pnl:+.1f}%' + ')</span>' if total_positions > 0 else '')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # 🚨 SOP Step 1: 先处理风险（红色区域，永远在最上面）
    # ============================================
    if has_risk:
        # 持仓紧急警报
        high_alerts = [a for a in position_alerts if a['urgency'] == 'high']
        if high_alerts:
            st.error(f"🚨 **紧急**: {len(high_alerts)} 只股票需要立即处理!")
            for alert in high_alerts:
                with st.container():
                    c1, c2, c3 = st.columns([2, 5, 2])
                    with c1:
                        st.markdown(f"### {alert['symbol']}")
                    with c2:
                        st.warning(f"⚠️ {alert['message']}")
                    with c3:
                        if st.button(f"🔴 {alert['action']}", key=f"sop_urgent_{alert['symbol']}", type="primary"):
                            st.session_state[f"show_detail_{alert['symbol']}"] = True
                    if st.session_state.get(f"show_detail_{alert['symbol']}"):
                        render_unified_stock_detail(
                            symbol=alert['symbol'], market=market,
                            show_charts=True, show_chips=False, show_news=False, show_actions=False,
                            key_prefix=f"sop_urgent_{alert['symbol']}"
                        )

    # === 空头信号（LIRED/PINK 逃顶预警）— 高优先级，默认展开 ===
    if bearish_alerts:
        high_urgency = [b for b in bearish_alerts if b['urgency'] == 'high']
        header_text = f"🩷 逃顶预警 ({len(bearish_alerts)} 只)" + (f" · ⚡{len(high_urgency)} 只高危" if high_urgency else "")
        with st.expander(header_text, expanded=has_risk):
            # 按 PINK 值排序
            sorted_bears = sorted(bearish_alerts, key=lambda x: x['pink_val'], reverse=True)
            
            # 构建表格数据
            table_rows = []
            for ba in sorted_bears[:25]:
                pink_v = ba['pink_val']
                lired_v = ba['lired_val']
                blue_d = ba['blue_daily']
                adx_v = ba['adx']
                vol_v = ba['volatility']
                
                # PINK 超买度 — 颜色标签
                if pink_v >= 95:
                    pink_tag = f'🔴 {pink_v:.0f}'
                elif pink_v > 90:
                    pink_tag = f'🟠 {pink_v:.0f}'
                elif pink_v > 80:
                    pink_tag = f'🟡 {pink_v:.0f}'
                else:
                    pink_tag = f'{pink_v:.0f}'
                
                # 信号类型
                sig_tags = []
                if pink_v > 90: sig_tags.append('PINK超买')
                if lired_v > 0: sig_tags.append('LIRED逃顶')
                if ba.get('urgency') == 'high' and not sig_tags: sig_tags.append('多空王卖')
                sig_str = ' + '.join(sig_tags) if sig_tags else '—'
                
                # BLUE 多空对比
                if blue_d > 100:
                    blue_tag = f'🟢 {blue_d:.0f}'  # 多头仍强，矛盾信号
                elif blue_d > 0:
                    blue_tag = f'🟡 {blue_d:.0f}'
                else:
                    blue_tag = '—'  # 无多头，空头确认
                
                # ADX 趋势强度
                if adx_v > 30:
                    adx_tag = f'💪 {adx_v:.0f}'
                elif adx_v > 20:
                    adx_tag = f'{adx_v:.0f}'
                else:
                    adx_tag = f'{adx_v:.0f}' if adx_v > 0 else '—'
                
                # 波动率
                vol_tag = f'{vol_v:.0%}' if vol_v > 0 else '—'
                
                name = ba['company_name'][:10] if ba['company_name'] else ''
                table_rows.append({
                    '股票': f"**{ba['symbol']}**",
                    '名称': name,
                    '价格': f"${ba['price']:.2f}" if ba['price'] > 0 else '—',
                    '信号': sig_str,
                    'PINK': pink_tag,
                    'BLUE日': blue_tag,
                    'ADX': adx_tag,
                    '波动': vol_tag,
                })
            
            if table_rows:
                import pandas as _pd
                bear_df = _pd.DataFrame(table_rows)
                st.dataframe(
                    bear_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        '股票': st.column_config.TextColumn('股票', width='small'),
                        '名称': st.column_config.TextColumn('名称', width='small'),
                        '价格': st.column_config.TextColumn('价格', width='small'),
                        '信号': st.column_config.TextColumn('信号类型', width='medium'),
                        'PINK': st.column_config.TextColumn('PINK 超买', width='small'),
                        'BLUE日': st.column_config.TextColumn('BLUE 多头', width='small'),
                        'ADX': st.column_config.TextColumn('ADX 趋势', width='small'),
                        '波动': st.column_config.TextColumn('波动率', width='small'),
                    }
                )
                st.caption("🔴 PINK≥95 极度超买 | 🟠 PINK 91-94 超买 | 🟡 PINK 81-90 高位 | 🟢 BLUE多头仍强(矛盾信号) | 💪 ADX>30 强趋势")

    # ============================================
    # 📈 信号质量总览（按需加载，避免阻塞 tabs 渲染）
    # ============================================
    _quality_key = f"quality_computed_{market}"
    try:
      is_quality_expanded = st.session_state.get(_quality_key, False)
      if not is_quality_expanded:
          _run_quality = st.button("🔍 点击加载「信号质量分析大盘」 (需要1-3秒)", key=f"run_quality_{market}", use_container_width=True)
          if _run_quality:
              st.session_state[_quality_key] = True
              st.rerun()
      else:
        with st.expander(f"📈 信号质量总览（{action_window_label}）", expanded=True):
          if tracking_rows_for_action:
            # 统一口径: 全量历史时不再固定 360 天 / 20000 样本，避免“数据已补齐但统计不增长”
            quality_days_back = int(action_days_back) if int(action_days_back) > 0 else 0
            if tracking_rows_for_action:
                # 上限 20000：超过此数量 _prefetch_price_series_batch 会读取数百 MB
                # 价格历史，且缺失 symbol 会逐个调用 Polygon API，导致超时/OOM
                quality_max_rows = min(int(len(tracking_rows_for_action)), 20000)
            else:
                quality_max_rows = 0

            # 统一口径：三张表都基于同一交易事实样本（同一平仓规则）
            preview_exit_rule = st.session_state.get(f"action_exit_rule_{market}", "fixed_10d")
            preview_tp = float(st.session_state.get(f"action_rule_tp_{market}", 10))
            preview_sl = float(st.session_state.get(f"action_rule_sl_{market}", 6))
            preview_hold = int(st.session_state.get(f"action_rule_hold_{market}", 20))
            facts_df = _build_unified_trade_facts(
                rows=tracking_rows_for_action,
                exit_rule=preview_exit_rule,
                take_profit_pct=preview_tp,
                stop_loss_pct=preview_sl,
                max_hold_days=preview_hold,
                max_rows=max(1, int(quality_max_rows)),
            )
            min_samples_quality = 12
            min_samples_combo = 3
            q1, q2, q3 = st.columns(3)
            with q1:
                combo_df = pd.DataFrame()
                st.markdown("**🏆 最优策略组合 Top 10**")
                if not facts_df.empty:
                    combo_key = "combo_bucket_full" if "combo_bucket_full" in facts_df.columns else "combo_bucket"
                    
                    # 缩尾均值函数：用 1%/99% 分位 clamp，避免仙股暴涨拉高均值
                    def _winsorized_mean(s):
                        arr = s.values
                        if len(arr) >= 20:
                            lo, hi = np.percentile(arr, [1, 99])
                            arr = np.clip(arr, lo, hi)
                        return float(np.mean(arr))
                    
                    combo_df_all = (
                        facts_df.groupby(combo_key, as_index=False)
                        .agg(
                            样本数=("ret", "count"),
                            当前胜率=("win", lambda x: float(np.mean(x) * 100.0)),
                            当前平均收益=("ret", _winsorized_mean),
                            中位收益=("ret", "median"),
                        )
                    )
                    combo_df_all = combo_df_all[combo_df_all["样本数"] >= min_samples_combo]
                    # Top 10: 样本≥10 + 按胜率排序
                    min_top10_samples = 10
                    combo_df_top10_pool = combo_df_all[combo_df_all["样本数"] >= min_top10_samples].copy()
                    if combo_df_top10_pool.empty:
                        combo_df_top10_pool = combo_df_all.copy()
                    combo_df_top10_pool = combo_df_top10_pool.sort_values(["当前胜率", "当前平均收益", "样本数"], ascending=[False, False, False])
                    combo_df_top10_pool["当前胜率"] = combo_df_top10_pool["当前胜率"].round(1)
                    combo_df_top10_pool["当前平均收益"] = combo_df_top10_pool["当前平均收益"].round(2)
                    combo_df_top10_pool["中位收益"] = combo_df_top10_pool["中位收益"].round(2)
                    combo_df_top10 = combo_df_top10_pool.head(10)
                    combo_df = combo_df_top10.copy()
                    # 全量也排好序（给展开更多用）
                    combo_df_all = combo_df_all.sort_values(["当前胜率", "当前平均收益", "样本数"], ascending=[False, False, False])
                    combo_df_all["当前胜率"] = combo_df_all["当前胜率"].round(1)
                    combo_df_all["当前平均收益"] = combo_df_all["当前平均收益"].round(2)
                    combo_df_all["中位收益"] = combo_df_all["中位收益"].round(2)
                    if not combo_df_top10.empty:
                        total_sample = int(len(facts_df))
                        shown_sample_top = int(combo_df_top10["样本数"].sum())
                        st.caption(f"样本≥{min_top10_samples} · 按胜率排序 · Top 10 / 共 {len(combo_df_all)} 个组合 · 覆盖 {shown_sample_top}/{total_sample}")
                        st.dataframe(
                            combo_df_top10.rename(columns={combo_key: "组合", "当前胜率": "胜率(%)", "当前平均收益": "缩尾均收(%)", "中位收益": "中位收益(%)"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                        # 展开更多
                        if len(combo_df_all) > 10:
                            with st.expander(f"📋 展开更多组合 (共 {len(combo_df_all)} 个)", expanded=False):
                                combo_sort_mode = st.radio(
                                    "组合排序",
                                    options=["按胜率(选优)", "按样本(核对全量)"],
                                    horizontal=True,
                                    key=f"combo_sort_mode_{market}",
                                )
                                combo_limit = st.slider(
                                    "展示组合数",
                                    min_value=20,
                                    max_value=min(500, len(combo_df_all)),
                                    value=min(120, len(combo_df_all)),
                                    step=10,
                                    key=f"combo_limit_{market}",
                                )
                                if combo_sort_mode == "按样本(核对全量)":
                                    combo_df_full = combo_df_all.sort_values(["样本数", "当前胜率", "当前平均收益"], ascending=[False, False, False]).head(int(combo_limit))
                                else:
                                    combo_df_full = combo_df_all.head(int(combo_limit))
                                combo_df = combo_df_full.copy()  # 更新引用
                                shown_sample_full = int(combo_df_full["样本数"].sum())
                                coverage = (shown_sample_full / total_sample * 100.0) if total_sample > 0 else 0.0
                                st.caption(f"样本覆盖: {shown_sample_full} / {total_sample} ({coverage:.1f}%)")
                                st.dataframe(
                                    combo_df_full.rename(columns={combo_key: "组合", "当前胜率": "胜率(%)", "当前平均收益": "缩尾均收(%)", "中位收益": "中位收益(%)"}),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                    else:
                        st.info("组合样本不足")
                else:
                    st.info("统一交易样本不足")
                # 宽松组合独立展示：不并入严格组合统计
                if not facts_df.empty and "combo_bucket_relaxed" in facts_df.columns:
                    with st.expander("宽松组合（独立统计，不并入严格组合）", expanded=False):
                        relaxed_df = (
                            facts_df.groupby("combo_bucket_relaxed", as_index=False)
                            .agg(
                                样本数=("ret", "count"),
                                当前胜率=("win", lambda x: float(np.mean(x) * 100.0)),
                                当前平均收益=("ret", "mean"),
                            )
                        )
                        relaxed_df = relaxed_df[relaxed_df["样本数"] >= min_samples_combo]
                        relaxed_df = relaxed_df.sort_values(["当前胜率", "当前平均收益", "样本数"], ascending=[False, False, False]).head(30)
                        if not relaxed_df.empty:
                            relaxed_df["当前胜率"] = relaxed_df["当前胜率"].round(1)
                            relaxed_df["当前平均收益"] = relaxed_df["当前平均收益"].round(2)
                            st.dataframe(
                                relaxed_df.rename(columns={"combo_bucket_relaxed": "组合", "当前胜率": "当前胜率(%)", "当前平均收益": "当前平均收益(%)"}),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("宽松组合样本不足")

            with q2:
                st.markdown("**策略 × 市值（明细口径，按胜率）**")
                if not facts_df.empty:
                    strat_key = "strategy_bucket_full" if "strategy_bucket_full" in facts_df.columns else "strategy_bucket"
                    grp = (
                        facts_df.groupby([strat_key, "cap_category"], as_index=False)
                        .agg(
                            样本数=("ret", "count"),
                            胜率=("win", lambda x: float(np.mean(x) * 100.0)),
                            平均收益=("ret", "mean"),
                        )
                    )
                    grp = grp[grp["样本数"] >= min_samples_combo]
                    grp = grp.sort_values(["胜率", "平均收益", "样本数"], ascending=[False, False, False]).head(30)
                    if not grp.empty:
                        grp["胜率"] = grp["胜率"].round(1)
                        grp["平均收益"] = grp["平均收益"].round(2)
                        st.dataframe(
                            grp.rename(columns={strat_key: "策略", "cap_category": "市值层", "胜率": "胜率(%)", "平均收益": "平均收益(%)"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("策略×市值样本不足")
                else:
                    st.info("统一交易样本不足")

            with q3:
                ind_df = pd.DataFrame()
                st.markdown("**板块/行业（按胜率）**")
                if not facts_df.empty:
                    ind_df = (
                        facts_df.groupby("industry", as_index=False)
                        .agg(
                            样本数=("ret", "count"),
                            胜率=("win", lambda x: float(np.mean(x) * 100.0)),
                            平均收益=("ret", "mean"),
                        )
                    )
                    ind_df = ind_df[ind_df["样本数"] >= min_samples_quality]
                    ind_df = ind_df.sort_values(["胜率", "平均收益"], ascending=False).head(12)
                    if not ind_df.empty:
                        ind_df["胜率"] = ind_df["胜率"].round(1)
                        ind_df["平均收益"] = ind_df["平均收益"].round(2)
                        st.dataframe(
                            ind_df.rename(columns={"industry": "分组", "胜率": "胜率(%)", "平均收益": "平均收益(%)"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("暂无行业统计")
                else:
                    st.info("统一交易样本不足")

            # 给出可执行建议（顶级交易员视角）
            top_combo_txt = "-"
            top_strat_cap_txt = "-"
            top_ind_txt = "-"
            try:
                if not combo_df.empty:
                    top_combo = combo_df.iloc[0]
                    top_combo_txt = f"{top_combo.get('组合', top_combo.get('combo_bucket', '-'))}"
                if 'grp' in locals() and not grp.empty:
                    top_sc = grp.iloc[0]
                    top_strat_cap_txt = f"{top_sc.get('策略', top_sc.get('strategy_bucket', '-'))}/{top_sc.get('市值层', top_sc.get('cap_category', '-'))}"
                if not ind_df.empty:
                    top_ind = ind_df.iloc[0]
                    top_ind_txt = f"{top_ind.get('分组', top_ind.get('industry', '-'))}"
            except Exception:
                pass
            st.success(f"今日优先级建议: 先做 `{top_combo_txt}` 组合，再优先 `{top_strat_cap_txt}`，并聚焦 `{top_ind_txt}` 板块。")
            st.caption(
                f"统一口径: rule={preview_exit_rule}, TP={preview_tp:.0f}%, SL={preview_sl:.0f}%, Hold={preview_hold}d, "
                f"样本={len(facts_df) if not facts_df.empty else 0}, 最小样本={min_samples_quality}"
            )

            st.markdown("### 🧪 规则平仓评估（交易口径）")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                exit_rule = st.selectbox(
                    "平仓规则",
                    options=["fixed_5d", "fixed_10d", "fixed_20d", "tp_sl_time", "kdj_dead_cross", "top_divergence_guard", "duokongwang_sell"],
                    index=1,
                    key=f"action_exit_rule_{market}",
                )
            with r2:
                rule_tp = st.slider("止盈(%)", min_value=3, max_value=25, value=10, step=1, key=f"action_rule_tp_{market}")
            with r3:
                rule_sl = st.slider("止损(%)", min_value=2, max_value=20, value=6, step=1, key=f"action_rule_sl_{market}")
            with r4:
                rule_max_hold = st.slider("最长持有天", min_value=5, max_value=60, value=20, step=1, key=f"action_rule_hold_{market}")
            rules_use_risk_params = {"tp_sl_time", "top_divergence_guard"}
            if exit_rule not in rules_use_risk_params:
                st.caption("当前规则不使用止盈/止损参数；仅 `tp_sl_time` 与 `top_divergence_guard` 使用这些参数。")

            # 优先读预计算表
            eval_ret = None
            try:
                from scripts.precompute_exit_results import get_precomputed_summary
                eval_ret = get_precomputed_summary(
                    market=market,
                    rule_name=exit_rule,
                    tp=float(rule_tp),
                    sl=float(rule_sl),
                    hold=int(rule_max_hold),
                )
                if not eval_ret or eval_ret.get("sample", 0) == 0:
                    eval_ret = None
            except Exception:
                eval_ret = None

            if eval_ret is None:
                eval_ret = evaluate_exit_rule(
                    rows=tracking_rows_for_action,
                    rule_name=exit_rule,
                    take_profit_pct=float(rule_tp),
                    stop_loss_pct=float(rule_sl),
                    max_hold_days=int(rule_max_hold),
                    max_rows=min(max(1, int(quality_max_rows)), 5000),
                )

            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("规则样本", int(eval_ret.get("sample") or 0))
            e2.metric("规则胜率", f"{float(eval_ret.get('win_rate_pct') or 0):.1f}%")
            e3.metric("规则均收", f"{float(eval_ret.get('avg_return_pct') or 0):+.2f}%")
            e4.metric("平均开始赚钱天数", f"{eval_ret.get('avg_first_profit_day') or '-'}")
            e5.metric("平均开始不赚钱天数", f"{eval_ret.get('avg_first_nonprofit_day') or '-'}")
            st.caption(f"平均盈利持续天数: {eval_ret.get('avg_profit_span_days') or '-'} 天（= 由盈转亏天 - 首次盈利天）")
            if int(eval_ret.get("sample") or 0) == 0:
                st.warning("当前规则评估样本为 0。请先在“组合追踪”里执行“回填历史扫描 + 刷新追踪”，再比较规则参数。")

            st.markdown("### 🧩 策略组合层（Meta Allocator）")
            st.caption(
                f"当前卖出规则口径: `{exit_rule}`"
                + (
                    f" (止盈{float(rule_tp):.0f}% / 止损{float(rule_sl):.0f}% / 最长持有{int(rule_max_hold)}天)"
                    if exit_rule in rules_use_risk_params
                    else " (技术规则/固定持有；止盈止损参数不参与)"
                )
            )
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                alloc_fee_bps = st.slider("手续费(bps)", min_value=0.0, max_value=30.0, value=5.0, step=0.5, key=f"alloc_fee_bps_{market}")
            with a2:
                alloc_slip_bps = st.slider("滑点(bps)", min_value=0.0, max_value=30.0, value=5.0, step=0.5, key=f"alloc_slip_bps_{market}")
            with a3:
                alloc_min_samples = st.slider("最小样本", min_value=4, max_value=120, value=12, step=2, key=f"alloc_min_samples_{market}")
            with a4:
                alloc_top_n = st.slider("当日候选数", min_value=5, max_value=30, value=12, step=1, key=f"alloc_topn_{market}")
            auto_best_exit = st.checkbox(
                "每个策略自动选择最优卖出规则",
                value=True,
                key=f"alloc_auto_best_exit_{market}",
            )

            if auto_best_exit:
                exit_rule_candidates = [
                    {"rule_name": "fixed_5d", "take_profit_pct": float(rule_tp), "stop_loss_pct": float(rule_sl), "max_hold_days": int(rule_max_hold)},
                    {"rule_name": "fixed_10d", "take_profit_pct": float(rule_tp), "stop_loss_pct": float(rule_sl), "max_hold_days": int(rule_max_hold)},
                    {"rule_name": "fixed_20d", "take_profit_pct": float(rule_tp), "stop_loss_pct": float(rule_sl), "max_hold_days": int(rule_max_hold)},
                    {"rule_name": "tp_sl_time", "take_profit_pct": float(rule_tp), "stop_loss_pct": float(rule_sl), "max_hold_days": int(rule_max_hold)},
                    {"rule_name": "kdj_dead_cross", "take_profit_pct": float(rule_tp), "stop_loss_pct": float(rule_sl), "max_hold_days": int(rule_max_hold)},
                    {"rule_name": "top_divergence_guard", "take_profit_pct": float(rule_tp), "stop_loss_pct": float(rule_sl), "max_hold_days": int(rule_max_hold)},
                    {"rule_name": "duokongwang_sell", "take_profit_pct": float(rule_tp), "stop_loss_pct": float(rule_sl), "max_hold_days": int(rule_max_hold)},
                ]
                perf_rows = evaluate_strategy_baskets_best_exit(
                    rows=tracking_rows_for_action,
                    exit_rule_candidates=exit_rule_candidates,
                    fee_bps=float(alloc_fee_bps),
                    slippage_bps=float(alloc_slip_bps),
                    min_samples=int(alloc_min_samples),
                    max_rows=max(1, int(quality_max_rows)),
                    primary_only=False,
                )
                st.caption("当前按“每个策略最优卖出规则”排序与分配权重。")
            else:
                perf_rows = evaluate_strategy_baskets(
                    rows=tracking_rows_for_action,
                    rule_name=exit_rule,
                    take_profit_pct=float(rule_tp),
                    stop_loss_pct=float(rule_sl),
                    max_hold_days=int(rule_max_hold),
                    fee_bps=float(alloc_fee_bps),
                    slippage_bps=float(alloc_slip_bps),
                    min_samples=int(alloc_min_samples),
                    max_rows=max(1, int(quality_max_rows)),
                    primary_only=False,
                )
            display_perf_rows = list(perf_rows or [])
            if len(display_perf_rows) < 4 and tracking_rows_for_action:
                # 展示口径兜底：即使样本偏少，也尽量展示更多策略行供参考
                try:
                    if auto_best_exit:
                        display_perf_rows = evaluate_strategy_baskets_best_exit(
                            rows=tracking_rows_for_action,
                            exit_rule_candidates=exit_rule_candidates,
                            fee_bps=float(alloc_fee_bps),
                            slippage_bps=float(alloc_slip_bps),
                            min_samples=1,
                            max_rows=max(1, int(quality_max_rows)),
                            primary_only=False,
                        )
                    else:
                        display_perf_rows = evaluate_strategy_baskets(
                            rows=tracking_rows_for_action,
                            rule_name=exit_rule,
                            take_profit_pct=float(rule_tp),
                            stop_loss_pct=float(rule_sl),
                            max_hold_days=int(rule_max_hold),
                            fee_bps=float(alloc_fee_bps),
                            slippage_bps=float(alloc_slip_bps),
                            min_samples=1,
                            max_rows=max(1, int(quality_max_rows)),
                            primary_only=False,
                        )
                except Exception:
                    display_perf_rows = list(perf_rows or [])

            perf_df = pd.DataFrame(display_perf_rows) if display_perf_rows else pd.DataFrame()
            if not perf_df.empty:
                show_cols = [
                    "策略", "sample", "net_win_rate_pct", "net_avg_return_pct",
                    "total_return_pct", "ann_return_pct", "ann_return_raw_pct", "ann_return_expect_pct",
                    "sharpe", "max_drawdown_pct",
                    "profit_factor", "turnover_per_year", "meta_score",
                ]
                if "exit_rule_desc" in perf_df.columns:
                    show_cols.extend(["exit_rule", "exit_rule_desc"])
                col_map = {
                    "sample": "样本",
                    "net_win_rate_pct": "净胜率(%)",
                    "net_avg_return_pct": "单笔净均收(%)",
                    "total_return_pct": "总收益(%)",
                    "ann_return_pct": "几何年化(主,%)",
                    "ann_return_raw_pct": "原始几何年化(%)",
                    "ann_return_expect_pct": "期望年化(均值,%)",
                    "sharpe": "Sharpe",
                    "max_drawdown_pct": "最大回撤(%)",
                    "profit_factor": "盈亏比",
                    "turnover_per_year": "年换手(次)",
                    "meta_score": "组合评分",
                    "exit_rule": "最优卖出规则",
                    "exit_rule_desc": "规则参数",
                }
                st.dataframe(
                    perf_df[show_cols].rename(columns=col_map),
                    use_container_width=True,
                    hide_index=True,
                )
                if "ann_return_raw_pct" in perf_df.columns:
                    st.caption("解释: 主口径=截尾后的几何复利年化；原始几何年化不截尾；期望年化为均值法，仅作对照。")

                weight_source_rows = list(perf_rows or [])
                if not weight_source_rows:
                    weight_source_rows = [r for r in display_perf_rows if int(r.get("sample", 0) or 0) >= 8]
                if not weight_source_rows:
                    weight_source_rows = list(display_perf_rows[:3])

                weight_rows = allocate_meta_weights(weight_source_rows, max_weight=0.45, min_weight=0.05)
                weight_df = pd.DataFrame(weight_rows) if weight_rows else pd.DataFrame()
                b1, b2 = st.columns([1, 1])
                with b1:
                    st.markdown("**动态权重建议**")
                    if not weight_df.empty:
                        st.dataframe(weight_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("暂无可分配策略")
                with b2:
                    st.markdown("**当日组合候选（按权重）**")
                    candidate_mode = st.radio(
                        "候选池口径",
                        options=["仅当日新信号", "含历史延续信号"],
                        horizontal=True,
                        key=f"alloc_candidate_mode_{market}",
                    )
                    exec_capital = st.number_input(
                        "执行资金($)",
                        min_value=1000.0,
                        max_value=5000000.0,
                        value=100000.0,
                        step=1000.0,
                        key=f"alloc_exec_capital_{market}",
                    )
                    history_days = 20
                    if candidate_mode == "含历史延续信号":
                        history_days = st.slider(
                            "历史延续窗口(天)",
                            min_value=3,
                            max_value=60,
                            value=20,
                            step=1,
                            key=f"alloc_history_days_{market}",
                        )
                    # 兼容云端旧缓存：meta_allocator_service 可能尚未包含新参数
                    try:
                        today_plan = build_today_meta_plan(
                            rows=tracking_rows_for_action,
                            weight_rows=weight_rows,
                            top_n=int(alloc_top_n),
                            total_capital=float(exec_capital),
                            include_history=(candidate_mode == "含历史延续信号"),
                            max_signal_age_days=int(history_days),
                        )
                    except TypeError:
                        try:
                            today_plan = build_today_meta_plan(
                                rows=tracking_rows_for_action,
                                weight_rows=weight_rows,
                                top_n=int(alloc_top_n),
                                total_capital=float(exec_capital),
                            )
                        except TypeError:
                            today_plan = build_today_meta_plan(
                                rows=tracking_rows_for_action,
                                weight_rows=weight_rows,
                                top_n=int(alloc_top_n),
                            )
                    today_plan_df = pd.DataFrame(today_plan) if today_plan else pd.DataFrame()
                    if not today_plan_df.empty:
                        if candidate_mode == "仅当日新信号":
                            show_cols_plan = [
                                "信号日期",
                                "symbol",
                                "主策略",
                                "命中策略数",
                                "命中策略",
                                "策略权重合计(%)",
                                "综合执行分(0-100)",
                                "建议仓位(%)",
                                "建议金额($)",
                                "买入触发价",
                                "失效价",
                                "R值",
                                "第一止盈价(2R)",
                                "执行状态",
                                "blue_daily",
                                "blue_weekly",
                            ]
                        else:
                            show_cols_plan = [
                                "信号日期",
                                "symbol",
                                "主策略",
                                "距信号天数",
                                "命中策略数",
                                "命中策略",
                                "策略权重合计(%)",
                                "综合执行分(0-100)",
                                "建议仓位(%)",
                                "建议金额($)",
                                "信号价",
                                "现价",
                                "价格变化($)",
                                "价格变化(%)",
                                "买入触发价",
                                "失效价",
                                "R值",
                                "第一止盈价(2R)",
                                "执行状态",
                                "blue_daily",
                                "blue_weekly",
                            ]
                        today_plan_df = today_plan_df[[c for c in show_cols_plan if c in today_plan_df.columns]]
                        st.dataframe(today_plan_df, use_container_width=True, hide_index=True)
                        if candidate_mode == "仅当日新信号":
                            st.caption("口径: 仅展示最新扫描日的新信号执行池。综合执行分=策略权重合计×信号强度归一化。")
                        else:
                            st.caption("口径: 历史延续跟踪池。价格变化=现价相对信号价变化；建议仓位/金额按候选内部相对分数分配。")
                        st.caption("执行定义: 买入触发价=信号突破确认价；失效价=风控退出线；第一止盈价(2R)=入场后首个减仓参考位。")
                    else:
                        st.info("今日暂无满足组合规则的候选。")
            else:
                st.info("组合层样本不足：请先积累更多候选追踪样本。")

            st.markdown("### 🔬 极致条件提升（Lift）")
            lift_ret = _analyze_extreme_lift(
                market=market,
                days_back=quality_days_back,
                exit_rule=exit_rule,
                take_profit_pct=float(rule_tp),
                stop_loss_pct=float(rule_sl),
                max_hold_days=int(rule_max_hold),
                max_rows=max(1, int(quality_max_rows)),
                schema_ver=2,
            )
            if lift_ret.get("ok"):
                lift_df = pd.DataFrame(lift_ret.get("table") or [])
                if not lift_df.empty:
                    st.dataframe(lift_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "说明: 胜率提升=组合胜率-基线胜率；"
                        f"收益口径={lift_ret.get('target_col')}（规则={lift_ret.get('rule_name') or exit_rule}）。"
                        "建议关注样本数，样本<20仅作参考。"
                    )
                    combo_details = lift_ret.get("combo_details") or {}
                    combo_options = [str(x) for x in lift_df["组合"].tolist() if str(x) in combo_details]
                    if combo_options:
                        st.markdown("**点击查看组合明细（逐笔）**")
                        selected_combo = st.selectbox(
                            "选择组合",
                            options=combo_options,
                            key=f"lift_combo_select_{market}",
                        )
                        detail_rows = list(combo_details.get(selected_combo) or [])
                        if detail_rows:
                            detail_df = pd.DataFrame(detail_rows)
                            win_cnt = int((detail_df["is_win"] == 1).sum()) if "is_win" in detail_df.columns else 0
                            sample_cnt = int(len(detail_df))
                            wr_calc = (win_cnt / sample_cnt * 100.0) if sample_cnt > 0 else 0.0
                            st.caption(
                                f"胜率计算: 胜率 = 盈利笔数 / 样本数 = {win_cnt}/{sample_cnt} = {wr_calc:.1f}%"
                            )
                            show_cols = [
                                "symbol",
                                "signal_date",
                                "signal_price",
                                "current_price",
                                "exit_day",
                                "exit_return_pct",
                                "is_win",
                                "signal_tags",
                                "blue_daily",
                                "blue_weekly",
                                "blue_monthly",
                            ]
                            detail_df = detail_df[[c for c in show_cols if c in detail_df.columns]]
                            detail_df = detail_df.rename(
                                columns={
                                    "symbol": "股票",
                                    "signal_date": "信号日期",
                                    "signal_price": "信号价",
                                    "current_price": "现价",
                                    "exit_day": "退出天数",
                                    "exit_return_pct": "规则收益(%)",
                                    "is_win": "是否盈利(1/0)",
                                    "signal_tags": "触发信号",
                                    "blue_daily": "日BLUE",
                                    "blue_weekly": "周BLUE",
                                    "blue_monthly": "月BLUE",
                                }
                            )
                            st.dataframe(detail_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("该组合暂无逐笔明细。")
                    else:
                        st.warning("当前缓存未包含逐笔明细，请点击侧边栏“🔄 刷新数据”后重试。")
                else:
                    st.info("暂无可用组合统计")
            else:
                st.info("极致条件统计暂无样本，请先在“组合追踪”完成回填与刷新。")
          else:
              st.info("暂无候选追踪样本，先运行扫描并回填历史。下面展示当日扫描替代视图。")
              if not df.empty:
                  fallback_cols = [
                      "symbol", "price", "blue_daily", "blue_weekly", "blue_monthly",
                      "heima_daily", "heima_weekly", "heima_monthly",
                      "juedi_daily", "juedi_weekly", "juedi_monthly",
                      "cap_category", "industry",
                  ]
                  show_cols = [c for c in fallback_cols if c in df.columns]
                  if show_cols:
                      fallback_df = df[show_cols].copy().head(50)
                      fallback_df = fallback_df.rename(
                          columns={
                              "symbol": "股票",
                              "price": "现价",
                              "blue_daily": "日BLUE",
                              "blue_weekly": "周BLUE",
                              "blue_monthly": "月BLUE",
                              "heima_daily": "日黑马",
                              "heima_weekly": "周黑马",
                              "heima_monthly": "月黑马",
                              "juedi_daily": "日掘地",
                              "juedi_weekly": "周掘地",
                              "juedi_monthly": "月掘地",
                              "cap_category": "市值层",
                              "industry": "行业",
                          }
                      )
                      st.dataframe(fallback_df, use_container_width=True, hide_index=True)
    except Exception as _quality_err:
        st.warning(f"信号质量总览加载失败（不影响下方 Tab 功能）: {_quality_err}")
    
    st.divider()
    
    # ============================================
    # 📋 核心工作区 (Tabs) - SOP 驱动的 5-Tab 布局
    # ============================================
    work_tab1, work_tab2, work_tab3, work_tab4, work_tab5 = st.tabs([
        "⚡ 今日行动",
        "🔎 发现新股", 
        "🎯 策略精选",
        "💼 我的持仓",
        "🔥 热点板块",
    ])

    
    # === Tab 1: 今日行动 (SOP Step 2 — 买入执行) ===
    with work_tab1:
        # 轻量 SOP 提示条
        st.markdown("""
        <div style="display: flex; gap: 12px; margin-bottom: 12px;">
            <div style="flex: 1; background: rgba(255,82,82,0.08); border-left: 3px solid #FF5252; padding: 10px 14px; border-radius: 6px; font-size: 0.9em;">
                <b>①</b> 先处理红色卖出/止损 · 控回撤优先
            </div>
            <div style="flex: 1; background: rgba(0,200,83,0.08); border-left: 3px solid #00C853; padding: 10px 14px; border-radius: 6px; font-size: 0.9em;">
                <b>②</b> 只做 A/B 级信号 · 看胜率+样本数
            </div>
            <div style="flex: 1; background: rgba(255,214,0,0.08); border-left: 3px solid #FFD600; padding: 10px 14px; border-radius: 6px; font-size: 0.9em;">
                <b>③</b> 单票分批进场 · 按建议仓位执行
            </div>
        </div>
        """, unsafe_allow_html=True)

        if tracking_rows_for_action:
            overall_win = sum(1 for r in tracking_rows_for_action if float(r.get("pnl_pct") or 0) > 0)
            overall_wr = overall_win / len(tracking_rows_for_action) * 100.0
            overall_avg = float(np.mean([float(r.get("pnl_pct") or 0) for r in tracking_rows_for_action]))
            st.caption(
                f"历史基准（{market}, {action_window_label}）: 样本 {len(tracking_rows_for_action)} | "
                f"胜率 {overall_wr:.1f}% | 平均收益 {overall_avg:+.2f}%"
            )
        
        # 两列布局：左边买入机会，右边其他任务
        action_left, action_right = st.columns([1, 1])
        
        with action_left:
            st.markdown("### 🟢 今日买入机会")
            pz1, pz2, pz3, pz4 = st.columns(4)
            with pz1:
                precision_mode = st.checkbox("高胜率模式", value=True, key=f"action_precision_{market}")
            with pz2:
                min_hist_samples = st.slider("最小样本", min_value=5, max_value=80, value=15, step=1, key=f"action_min_samples_{market}")
            with pz3:
                min_hist_win = st.slider("最小胜率(%)", min_value=45, max_value=80, value=56, step=1, key=f"action_min_win_{market}")
            with pz4:
                min_hist_avg = st.slider("最小均收(%)", min_value=-1.0, max_value=5.0, value=0.5, step=0.1, key=f"action_min_avg_{market}")
            
            # 获取强势信号
            if not df.empty and 'blue_daily' in df.columns:
                strong_raw = df[
                    (df['blue_daily'].fillna(0) > 100) & 
                    (df['blue_weekly'].fillna(0) > 50)
                ].copy()
                strong_raw = strong_raw.sort_values(
                    by=["blue_daily", "blue_weekly"],
                    ascending=False,
                ).head(30)

                strong_candidates = []
                for _, row in strong_raw.iterrows():
                    tags = derive_signal_tags(dict(row))
                    rel = _calc_signal_reliability(tags)
                    if precision_mode:
                        wr = float(rel.get("win_rate") or 0.0)
                        avg = float(rel.get("avg_pnl") or 0.0)
                        if rel.get("sample", 0) < int(min_hist_samples):
                            continue
                        if wr < float(min_hist_win):
                            continue
                        if avg < float(min_hist_avg):
                            continue
                        if rel.get("grade") not in ("A", "B"):
                            continue
                    strong_candidates.append((row, tags, rel))
                strong = strong_candidates[:8]
                
                if strong:
                    for row, tags, rel in strong:
                        symbol = row.get('symbol', '')
                        company_name = row.get('company_name', '')
                        blue_d = row.get('blue_daily', 0)
                        blue_w = row.get('blue_weekly', 0)
                        signal_price = float(row.get('price', 0) or 0)
                        current_price = signal_price
                        try:
                            hist_px = _cached_stock_data(symbol, market=market, days=10)
                            if hist_px is not None and not hist_px.empty and 'Close' in hist_px.columns:
                                current_price = float(hist_px['Close'].iloc[-1] or signal_price)
                        except Exception:
                            current_price = signal_price
                        price_change_pct = ((current_price / signal_price - 1.0) * 100.0) if signal_price > 0 else 0.0
                        signal_labels = []
                        if bool(row.get('heima_daily') or row.get('is_heima')):
                            signal_labels.append('日黑马')
                        if bool(row.get('heima_weekly')):
                            signal_labels.append('周黑马')
                        if bool(row.get('heima_monthly')):
                            signal_labels.append('月黑马')
                        if bool(row.get('juedi_daily') or row.get('is_juedi')):
                            signal_labels.append('日掘地')
                        if bool(row.get('juedi_weekly')):
                            signal_labels.append('周掘地')
                        if bool(row.get('juedi_monthly')):
                            signal_labels.append('月掘地')
                        signal_text = "、".join(signal_labels) if signal_labels else "无特殊信号"
                        
                        # 价格符号和名称显示
                        price_sym = "¥" if market == "CN" else "$"
                        display_name = company_name if company_name else symbol
                        display_code = symbol.split('.')[0] if '.' in symbol else symbol
                        
                        # 卡片式展示
                        with st.container():
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1a472a22, #1a472a11); 
                                        border-left: 3px solid #00C853; padding: 12px; 
                                        border-radius: 8px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.1em; font-weight: bold;">{display_name}</span>
                                        <span style="font-size: 0.8em; color: #888; margin-left: 4px;">{display_code}</span>
                                    </div>
                                    <span style="color: #00C853;">现价 {price_sym}{current_price:.2f}</span>
                                </div>
                                <div style="font-size: 0.9em; color: #888; margin-top: 4px;">
                                    信号价 {price_sym}{signal_price:.2f} → 现价 {price_sym}{current_price:.2f} ({price_change_pct:+.2f}%)
                                </div>
                                <div style="font-size: 0.9em; color: #888; margin-top: 2px;">
                                    日BLUE {blue_d:.0f} | 周BLUE {blue_w:.0f}
                                </div>
                                <div style="font-size: 0.9em; color: #BDBDBD; margin-top: 2px;">
                                    信号类型: {signal_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            wr_txt = f"{rel['win_rate']:.1f}%" if rel['win_rate'] is not None else "-"
                            avg_txt = f"{rel['avg_pnl']:+.2f}%" if rel['avg_pnl'] is not None else "-"
                            st.caption(
                                f"评级 {rel['grade']} | 样本 {rel['sample']} | 胜率 {wr_txt} | 均收 {avg_txt} | 建议: {rel['position_hint']}"
                            )
                            
                            # 操作按钮
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button("📊 查看详情", key=f"view_{symbol}", use_container_width=True):
                                    st.session_state['action_selected_symbol'] = symbol
                            with btn_col2:
                                if st.button("💰 模拟买入", key=f"buy_{symbol}", use_container_width=True):
                                    st.session_state['action_buy_symbol'] = symbol
                else:
                    if precision_mode:
                        st.info("高胜率模式下暂无达标信号。可降低阈值或切换普通模式。")
                    else:
                        st.info("今日暂无强势买入信号")
            else:
                st.info("正在加载数据...")
        
        with action_right:
            # === 逃顶预警 (幻影主力) ===
            st.markdown("### 🚨 逃顶 & 买入预警")
            try:
                # 使用缓存函数，避免每次交互重算 30 只股票的指标
                scan_symbols = tuple(df['symbol'].tolist()[:30]) if not df.empty and 'symbol' in df.columns else ()
                
                if scan_symbols:
                    escape_warnings, golden_bottoms, trend_pullbacks = _cached_escape_warning_scan(
                        symbols=scan_symbols, market=market
                    )
                else:
                    escape_warnings, golden_bottoms, trend_pullbacks = [], [], []
                
                if escape_warnings:
                    for ew in escape_warnings:
                        colors = {
                            'critical': ("#FF0000", "rgba(255,0,0,0.15)"),
                            'high': ("#FF1744", "rgba(255,23,68,0.12)"),
                            'low': ("#FF6D00", "rgba(255,109,0,0.08)")
                        }
                        border, bg = colors.get(ew['level'], colors['low'])
                        reason = ew.get('reason', '')
                        st.markdown(f"""
                        <div style="background: {bg}; border-left: 3px solid {border}; 
                                    padding: 10px; border-radius: 8px; margin-bottom: 6px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: bold;">{ew['name']}</span>
                                <span style="color: {border};">{'🚨' if ew['level'] == 'critical' else '⚠️'} {reason}</span>
                            </div>
                            <div style="font-size: 0.85em; color: #888;">
                                PINK: {ew['pink']:.1f} | {ew['price_sym']}{ew['price']:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ 暂无逃顶预警")
                
                # 黄金底 (最强买入信号)
                if golden_bottoms:
                    st.markdown("### 🎯 黄金底信号")
                    st.markdown("""
                    <div style="background: rgba(255,215,0,0.06); padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; font-size: 0.88em; color: #ccc;">
                        <b>什么是黄金底？</b> CCI极度超卖（<-100）+ 底部金叉 = 超跌反弹信号。回测胜率 69%。<br/>
                        <b>怎么做？</b> ① 加入观察列表 → ② 等放量确认（次日量比>1.5）→ ③ 试仓10-15% → ④ 止损设在信号价下方8%
                    </div>
                    """, unsafe_allow_html=True)
                    for gb in golden_bottoms:
                        stop_loss_price = gb['price'] * 0.92
                        target_price = gb['price'] * 1.15
                        price_sym = gb['price_sym']
                        st.markdown(f"""
                        <div style="background: rgba(255,215,0,0.12); border-left: 3px solid #FFD700;
                                    padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: bold; font-size: 1.05em;">{gb['name']}</span>
                                    <span style="color: #FFD700; margin-left: 6px;">🎯 黄金底</span>
                                </div>
                                <span style="font-size: 1.1em;">{price_sym}{gb['price']:.2f}</span>
                            </div>
                            <div style="font-size: 0.85em; color: #888; margin-top: 4px;">
                                CCI: {gb['cci']:.0f} · 止损 {price_sym}{stop_loss_price:.2f}(-8%) · 目标 {price_sym}{target_price:.2f}(+15%)
                            </div>
                            <div style="font-size: 0.82em; color: #FFD700; margin-top: 4px;">
                                👉 操作: 加入观察 → 等放量确认 → 试仓10%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        gb_sym = gb['symbol']
                        btn_col_gb1, btn_col_gb2 = st.columns(2)
                        with btn_col_gb1:
                            if st.button(f"📊 {gb_sym} 详情", key=f"gb_detail_{gb_sym}", use_container_width=True):
                                st.session_state['action_selected_symbol'] = gb_sym
                        with btn_col_gb2:
                            if st.button(f"💰 模拟买入", key=f"gb_buy_{gb_sym}", use_container_width=True):
                                st.session_state['action_buy_symbol'] = gb_sym
                
                # 趋势回调买入
                if trend_pullbacks:
                    st.markdown("### 📈 趋势回调买入")
                    st.markdown("""
                    <div style="background: rgba(0,200,83,0.06); padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; font-size: 0.88em; color: #ccc;">
                        <b>什么是趋势回调？</b> 海底捞月消失 + ADX>25 = 强趋势中的回调买点。回测胜率 61%。<br/>
                        <b>怎么做？</b> ① 确认ADX>25（趋势仍在）→ ② 在支撑位附近挂单 → ③ 仓位15-20% → ④ 止损设在前低下方5%
                    </div>
                    """, unsafe_allow_html=True)
                    for tp in trend_pullbacks:
                        stop_loss_tp = tp['price'] * 0.95
                        target_tp = tp['price'] * 1.12
                        price_sym_tp = tp['price_sym']
                        st.markdown(f"""
                        <div style="background: rgba(0,200,83,0.08); border-left: 3px solid #00C853;
                                    padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: bold; font-size: 1.05em;">{tp['name']}</span>
                                    <span style="color: #00C853; margin-left: 6px;">📈 回调买入</span>
                                </div>
                                <span style="font-size: 1.1em;">{price_sym_tp}{tp['price']:.2f}</span>
                            </div>
                            <div style="font-size: 0.85em; color: #888; margin-top: 4px;">
                                ADX: {tp['adx']:.0f} · 止损 {price_sym_tp}{stop_loss_tp:.2f}(-5%) · 目标 {price_sym_tp}{target_tp:.2f}(+12%)
                            </div>
                            <div style="font-size: 0.82em; color: #00C853; margin-top: 4px;">
                                👉 操作: 确认ADX>25 → 支撑位挂单 → 仓位15%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        tp_sym = tp['symbol']
                        btn_col_tp1, btn_col_tp2 = st.columns(2)
                        with btn_col_tp1:
                            if st.button(f"📊 {tp_sym} 详情", key=f"tp_detail_{tp_sym}", use_container_width=True):
                                st.session_state['action_selected_symbol'] = tp_sym
                        with btn_col_tp2:
                            if st.button(f"💰 模拟买入", key=f"tp_buy_{tp_sym}", use_container_width=True):
                                st.session_state['action_buy_symbol'] = tp_sym
            except Exception as e:
                st.caption(f"幻影主力扫描暂不可用: {e}")
            
            st.divider()
            
            st.markdown("### 📋 其他待办")
            
            # 观察列表提醒
            try:
                from services.signal_tracker import get_watchlist
                watchlist = get_watchlist(market=market)
                
                if watchlist:
                    st.markdown(f"**👁️ {len(watchlist)}只股票在观察中**")
                    for w in watchlist[:3]:
                        symbol = w.get('symbol', '')
                        entry = w.get('entry_price', 0)
                        st.markdown(f"- `{symbol}` 等待入场 @ ${entry:.2f}")
                    
                    if len(watchlist) > 3:
                        st.caption(f"...还有 {len(watchlist) - 3} 只")
            except:
                pass
            
            st.divider()
            
            # 中等优先级警报
            medium_alerts = [a for a in position_alerts if a['urgency'] in ['medium', 'low']]
            if medium_alerts:
                st.markdown("**⚠️ 持仓提醒**")
                for alert in medium_alerts[:3]:
                    icon = '🟡' if alert['urgency'] == 'medium' else '🟢'
                    st.markdown(f"{icon} **{alert['symbol']}**: {alert['message']}")
        
        # 显示选中的股票详情
        if st.session_state.get('action_selected_symbol'):
            st.divider()
            symbol = st.session_state['action_selected_symbol']
            st.markdown(f"### 📊 {symbol} 详细分析")
            
            # 关闭按钮
            if st.button("❌ 关闭详情", key="close_action_detail"):
                st.session_state['action_selected_symbol'] = None
                st.rerun()
            
            render_unified_stock_detail(
                symbol=symbol,
                market=market,
                key_prefix=f"tab1_action_{symbol}"
            )
        
        # 处理模拟买入
        if st.session_state.get('action_buy_symbol'):
            symbol = st.session_state['action_buy_symbol']
            st.divider()
            st.markdown(f"### 💰 模拟买入 {symbol}")
            
            with st.form(f"buy_form_{symbol}"):
                price = df[df['symbol'] == symbol]['price'].iloc[0] if symbol in df['symbol'].values else 100
                shares = st.number_input("买入股数", min_value=1, value=max(1, int(1000 / price)))
                stop_loss = st.number_input("止损价", value=price * 0.92)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("✅ 确认买入", type="primary"):
                        try:
                            from services.portfolio_service import paper_buy
                            result = paper_buy(symbol, shares, price, market)
                            if result.get('success'):
                                st.success(f"🎉 买入成功! {symbol} x {shares}股")
                                st.balloons()
                                st.session_state['action_buy_symbol'] = None
                            else:
                                st.error(result.get('error', '买入失败'))
                        except Exception as e:
                            st.error(f"买入失败: {e}")
                with col2:
                    if st.form_submit_button("❌ 取消"):
                        st.session_state['action_buy_symbol'] = None
                        st.rerun()
    
    # === Tab 2: 发现新股 (重新设计 - 卡片式浏览) ===
    with work_tab2:
        st.markdown("### 🔎 发现新股")
        
        # 筛选器（横向排列）
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            signal_filter = st.selectbox(
                "信号类型", 
                ["🔥 全部强信号", "📊 日线BLUE>100", "📈 日周共振", "🚀 日周月共振", "🐴 黑马信号"],
                key="discover_filter"
            )
        with filter_col2:
            sort_by = st.selectbox(
                "排序方式",
                ["日BLUE↓", "周BLUE↓", "月BLUE↓", "价格↓", "ADX↓"],
                key="discover_sort"
            )
        with filter_col3:
            show_count = st.slider("显示数量", 5, 30, 12, key="discover_count")
        
        st.divider()
        
        # 根据筛选条件过滤
        if not df.empty:
            filtered_df = df.copy()
            
            if signal_filter == "📊 日线BLUE>100":
                filtered_df = filtered_df[filtered_df['blue_daily'].fillna(0) > 100]
            elif signal_filter == "📈 日周共振":
                filtered_df = filtered_df[
                    (filtered_df['blue_daily'].fillna(0) > 100) & 
                    (filtered_df['blue_weekly'].fillna(0) > 80)
                ]
            elif signal_filter == "🚀 日周月共振":
                filtered_df = filtered_df[
                    (filtered_df['blue_daily'].fillna(0) > 100) & 
                    (filtered_df['blue_weekly'].fillna(0) > 80) &
                    (filtered_df['blue_monthly'].fillna(0) > 60)
                ]
            elif signal_filter == "🐴 黑马信号":
                # 检查黑马列
                heima_cols = [c for c in filtered_df.columns if 'heima' in c.lower()]
                if heima_cols:
                    heima_mask = filtered_df[heima_cols].apply(
                        lambda x: x.isin([True, 1, b'\x01']).any(), axis=1
                    )
                    filtered_df = filtered_df[heima_mask]
            
            # 排序
            sort_map = {
                "日BLUE↓": ('blue_daily', False),
                "周BLUE↓": ('blue_weekly', False),
                "月BLUE↓": ('blue_monthly', False),
                "价格↓": ('price', False),
                "ADX↓": ('adx', False)
            }
            sort_col, sort_asc = sort_map.get(sort_by, ('blue_daily', False))
            if sort_col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(sort_col, ascending=sort_asc)
            
            filtered_df = filtered_df.head(show_count)
            
            if filtered_df.empty:
                st.info("没有符合条件的股票")
            else:
                # 卡片式展示 (每行3个)
                st.markdown(f"**找到 {len(filtered_df)} 只股票** | 点击卡片查看详情")
                
                # 用session state记录选中的股票
                if 'discover_selected' not in st.session_state:
                    st.session_state['discover_selected'] = None
                
                # 使用columns展示卡片
                cols_per_row = 3
                for row_idx in range(0, len(filtered_df), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for col_idx, col in enumerate(cols):
                        data_idx = row_idx + col_idx
                        if data_idx >= len(filtered_df):
                            break
                        
                        row = filtered_df.iloc[data_idx]
                        symbol = row.get('symbol', 'N/A')
                        company_name = row.get('company_name', '')
                        price = row.get('price', 0)
                        blue_d = row.get('blue_daily', 0)
                        blue_w = row.get('blue_weekly', 0)
                        blue_m = row.get('blue_monthly', 0)
                        adx = row.get('adx', 0)
                        
                        # 价格符号
                        price_sym = "¥" if market == "CN" else "$"
                        
                        # 显示名称：有公司名则显示，否则只显示代码
                        display_name = f"{company_name}" if company_name else symbol
                        display_code = symbol.split('.')[0] if '.' in symbol else symbol  # 去掉 .SH/.SZ 后缀
                        
                        # 信号强度颜色
                        if blue_d > 100 and blue_w > 80:
                            card_color = "#00C853"  # 绿色 - 强信号
                            card_bg = "#1a472a"
                        elif blue_d > 100:
                            card_color = "#FFD600"  # 黄色 - 中等
                            card_bg = "#4a4a00"
                        else:
                            card_color = "#666"  # 灰色 - 弱
                            card_bg = "#333"
                        
                        with col:
                            # 卡片容器 - 显示名称和代码
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {card_bg}66, {card_bg}33); 
                                        border: 1px solid {card_color}44;
                                        border-radius: 12px; padding: 16px; margin-bottom: 12px;
                                        transition: transform 0.2s;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.2em; font-weight: bold; color: {card_color};">{display_name}</span>
                                        <span style="font-size: 0.85em; color: #888; margin-left: 6px;">{display_code}</span>
                                    </div>
                                    <span style="font-size: 1.1em;">{price_sym}{price:.2f}</span>
                                </div>
                                <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                                    <span style="background: #00C85333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        日B {blue_d:.0f}
                                    </span>
                                    <span style="background: #FFD60033; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        周B {blue_w:.0f}
                                    </span>
                                    <span style="background: #2196F333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        月B {blue_m:.0f}
                                    </span>
                                    <span style="background: #9C27B033; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        ADX {adx:.0f}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 操作按钮
                            if st.button(f"📊 {symbol} 详情", key=f"disc_{symbol}", use_container_width=True):
                                st.session_state['discover_selected'] = symbol
                
                # 显示选中股票的详情
                if st.session_state.get('discover_selected'):
                    st.divider()
                    symbol = st.session_state['discover_selected']
                    
                    # 关闭按钮
                    header_col1, header_col2 = st.columns([6, 1])
                    with header_col1:
                        st.markdown(f"### 📊 {symbol} 详细分析")
                    with header_col2:
                        if st.button("❌ 关闭", key="close_discover_detail"):
                            st.session_state['discover_selected'] = None
                            st.rerun()
                    
                    render_unified_stock_detail(
                        symbol=symbol,
                        market=market,
                        key_prefix=f"tab2_discover_{symbol}"
                    )
        else:
            st.info("正在加载数据...")
    
    # === Tab 3: 策略精选 (增强版 — 含历史胜率) ===
    with work_tab3:
        st.markdown("### 🎯 策略精选")
        st.caption("8大策略同时选股，多策略共识=高可信度 | 历史胜率来自候选追踪数据")
        
        # 获取策略选股
        manager = get_strategy_manager()
        all_picks = manager.get_all_picks(df, top_n=top_n)
        consensus = manager.get_consensus_picks(df, min_votes=2)
        
        # ── 策略 → 信号标签映射（用于从追踪数据计算历史胜率）──
        STRATEGY_TAG_MAP = {
            "momentum":        {"tags": ["DAY_BLUE"], "label": "🚀动量突破",    "desc": "BLUE>80+ADX>25"},
            "value":           {"tags": ["DAY_BLUE", "WEEK_BLUE"], "label": "💎价值洼地",  "desc": "日周BLUE共振+低波动"},
            "conservative":    {"tags": ["DAY_BLUE"], "label": "🛡️稳健保守",    "desc": "BLUE>60+高流动+低波动"},
            "aggressive":      {"tags": ["DAY_BLUE"], "label": "⚡激进突破",    "desc": "BLUE>90+ADX>30"},
            "multi_timeframe": {"tags": ["DAY_BLUE", "WEEK_BLUE"], "label": "🔄多周期共振", "desc": "日线+周线双BLUE"},
            "reversal":        {"tags": ["DAY_HEIMA"], "label": "🔃超跌反弹",   "desc": "绝地反击信号"},
            "volume_breakout": {"tags": ["DAY_BLUE"], "label": "📊放量突破",    "desc": "量价齐升+BLUE"},
            "heima":           {"tags": ["DAY_HEIMA"], "label": "🐴黑马形态",   "desc": "黑马底部形态"},
        }
        
        # 预计算每个策略的历史胜率
        strategy_perf_cache = {}
        if tracking_rows_for_action:
            for strat_key, strat_info in STRATEGY_TAG_MAP.items():
                required_tags = strat_info["tags"]
                matched = [
                    r for r in tracking_rows_for_action
                    if set(required_tags).issubset(set(r.get("signal_tags_list") or []))
                ]
                if matched:
                    wins = sum(1 for r in matched if float(r.get("pnl_pct") or 0) > 0)
                    avg_pnl = float(np.mean([float(r.get("pnl_pct") or 0) for r in matched]))
                    strategy_perf_cache[strat_key] = {
                        "win_rate": wins / len(matched) * 100,
                        "avg_pnl": avg_pnl,
                        "sample": len(matched),
                    }
                else:
                    strategy_perf_cache[strat_key] = {"win_rate": None, "avg_pnl": None, "sample": 0}
        
        # 构建策略显示名 → 策略key的反向映射
        label_to_key = {v["label"]: k for k, v in STRATEGY_TAG_MAP.items()}
        
        # 显示共识精选
        if consensus:
            st.markdown("#### 🏆 多策略共识 (被2个以上策略选中)")
            
            consensus_data = []
            for symbol, votes, avg_score, *rest in consensus[:10]:
                voted_strategies = rest[0] if rest else []
                stock_row = df[df['symbol'] == symbol].iloc[0] if not df.empty and symbol in df['symbol'].values else {}
                
                blue_d = stock_row.get('blue_daily', 0) if hasattr(stock_row, 'get') else (stock_row['blue_daily'] if 'blue_daily' in getattr(stock_row, 'index', []) else 0)
                blue_w = stock_row.get('blue_weekly', 0) if hasattr(stock_row, 'get') else (stock_row['blue_weekly'] if 'blue_weekly' in getattr(stock_row, 'index', []) else 0)
                price = stock_row.get('price', 0) if hasattr(stock_row, 'get') else (stock_row['price'] if 'price' in getattr(stock_row, 'index', []) else 0)
                
                # 计算该股票的信号可靠性
                tags = derive_signal_tags(dict(stock_row)) if hasattr(stock_row, 'get') or hasattr(stock_row, 'to_dict') else []
                rel = _calc_signal_reliability(tags)
                grade = rel.get('grade', 'N/A')
                wr_txt = f"{rel['win_rate']:.0f}%" if rel.get('win_rate') is not None else '-'
                avg_txt = f"{rel['avg_pnl']:+.1f}%" if rel.get('avg_pnl') is not None else '-'
                sample = rel.get('sample', 0)
                grade_icon = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '⚪'}.get(grade, '⚪')
                
                # 每个投票策略附带胜率
                strat_with_wr = []
                for sname in voted_strategies:
                    skey = label_to_key.get(sname, "")
                    perf = strategy_perf_cache.get(skey, {})
                    s_wr = perf.get("win_rate")
                    if s_wr is not None:
                        strat_with_wr.append(f"{sname}({s_wr:.0f}%)")
                    else:
                        strat_with_wr.append(sname)
                
                price_sym = "¥" if market == "CN" else "$"
                consensus_data.append({
                    '代码': symbol,
                    '⭐票数': votes,
                    '投票策略(胜率)': ', '.join(strat_with_wr) if strat_with_wr else '-',
                    '评级': f"{grade_icon}{grade}",
                    '信号胜率': wr_txt,
                    '信号均收': avg_txt,
                    '样本': sample,
                    '日BLUE': f"{blue_d:.0f}",
                    '周BLUE': f"{blue_w:.0f}",
                    '价格': f"{price_sym}{price:.2f}" if price else '-',
                })
            
            consensus_df = pd.DataFrame(consensus_data)
            
            event = st.dataframe(
                consensus_df,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )
            st.caption("🟢A=主仓40%+ | 🔵B=半仓20%+ | 🟡C=试仓10%+ | ⚪D=仅观察 | 策略胜率=该策略标签组合的历史追踪表现")
            
            selected_consensus_symbol = None
            if event and hasattr(event, 'selection') and event.selection.rows:
                idx = event.selection.rows[0]
                if idx < len(consensus_data):
                    selected_consensus_symbol = consensus_data[idx]['代码']
            
            if consensus_data:
                sel_col1, sel_col2 = st.columns([3, 1])
                with sel_col1:
                    selected_symbol = st.selectbox(
                        "选择股票加入观察",
                        [c['代码'] for c in consensus_data],
                        key="consensus_select"
                    )
                with sel_col2:
                    if st.button("📋 加入观察", key="add_consensus_watch", type="primary"):
                        try:
                            from services.signal_tracker import add_to_watchlist
                            sel_data = next((c for c in consensus_data if c['代码'] == selected_symbol), None)
                            if sel_data:
                                price_str = sel_data['价格'].replace('$', '').replace('¥', '')
                                price = float(price_str) if price_str != '-' else 0
                                add_to_watchlist(
                                    symbol=selected_symbol,
                                    market=market,
                                    entry_price=price,
                                    target_price=price * 1.15,
                                    stop_loss=price * 0.92,
                                    signal_type='consensus',
                                    signal_score=float(sel_data.get('⭐票数', 0)),
                                    notes=f"多策略共识 {sel_data['⭐票数']}票"
                                )
                                st.success(f"✅ {selected_symbol} 已加入观察列表")
                        except Exception as e:
                            st.error(f"添加失败: {e}")
            
            if selected_consensus_symbol:
                st.divider()
                st.markdown(f"### 📊 {selected_consensus_symbol} 详细分析")
                render_unified_stock_detail(
                    symbol=selected_consensus_symbol,
                    market=market,
                    key_prefix=f"tab3_consensus_{selected_consensus_symbol}"
                )
        else:
            st.info("暂无共识股票，请检查扫描数据")
        
        st.divider()
        
        # 策略总览表（使用标签映射计算真实胜率）
        if all_picks:
            st.markdown("#### 📊 策略历史表现总览")
            summary_rows = []
            for strategy_key, picks in all_picks.items():
                pick_count = len(picks or [])
                strat_info = STRATEGY_TAG_MAP.get(strategy_key, {})
                strat_label = strat_info.get("label", strategy_key)
                strat_desc = strat_info.get("desc", "")
                perf = strategy_perf_cache.get(strategy_key, {})
                
                s_wr = perf.get("win_rate")
                s_avg = perf.get("avg_pnl")
                s_sample = perf.get("sample", 0)
                
                summary_rows.append({
                    "策略": strat_label,
                    "选股逻辑": strat_desc,
                    "今日候选": pick_count,
                    "历史胜率": f"{s_wr:.0f}%" if s_wr is not None else '-',
                    "历史均收": f"{s_avg:+.1f}%" if s_avg is not None else '-',
                    "样本数": s_sample,
                })
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.caption("胜率=该策略对应标签组合在追踪数据中的历史表现。样本数越多越可信。")

            st.markdown("#### 📚 各策略明细")
            for strategy_name, picks in all_picks.items():
                with st.expander(f"{strategy_name} ({len(picks or [])}只)", expanded=False):
                    if not picks:
                        st.info("该策略当前无候选")
                        continue
                    normalized_picks = []
                    for p in picks:
                        if isinstance(p, dict):
                            normalized_picks.append(p)
                        else:
                            normalized_picks.append(
                                {
                                    "symbol": getattr(p, "symbol", ""),
                                    "price": getattr(p, "price", 0.0),
                                    "score": getattr(p, "score", 0.0),
                                    "stop_loss": getattr(p, "stop_loss", 0.0),
                                }
                            )
                    rows_show = []
                    for p in normalized_picks:
                        sym = str(p.get("symbol", "") or "")
                        price = float(p.get("price", 0.0) or 0.0)
                        score = float(p.get("score", 0.0) or 0.0)
                        stop_loss = float(p.get("stop_loss", 0.0) or 0.0)
                        # 每只候选加上历史可靠性
                        stock_row_for_tag = df[df['symbol'] == sym].iloc[0] if not df.empty and sym in df['symbol'].values else {}
                        pick_tags = derive_signal_tags(dict(stock_row_for_tag)) if hasattr(stock_row_for_tag, 'get') or hasattr(stock_row_for_tag, 'to_dict') else []
                        pick_rel = _calc_signal_reliability(pick_tags)
                        pick_grade = pick_rel.get('grade', '-')
                        pick_wr = f"{pick_rel['win_rate']:.0f}%" if pick_rel.get('win_rate') is not None else '-'
                        rows_show.append({
                            "代码": sym,
                            "评级": pick_grade,
                            "胜率": pick_wr,
                            "评分": round(score, 2),
                            "现价": round(price, 2) if price > 0 else None,
                            "止损价": round(stop_loss, 2) if stop_loss > 0 else None,
                        })
                    st.dataframe(pd.DataFrame(rows_show), use_container_width=True, hide_index=True)
        else:
            st.warning("当前策略列表为空。请检查扫描数据是否已加载，或切换市场后重试。")
    
    # === Tab 4: 我的持仓 (重新设计 - 专注持仓管理) ===
    with work_tab4:
        st.markdown("### 💼 我的持仓")
        
        # 持仓概览
        total_value = portfolio.get('total_value', 0)
        total_pnl = portfolio.get('total_pnl_pct', 0)
        cash = portfolio.get('cash', 100000)
        
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.metric("总资产", f"${total_value + cash:,.0f}")
        with p_col2:
            st.metric("持仓市值", f"${total_value:,.0f}")
        with p_col3:
            delta_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("总盈亏", f"{total_pnl:+.1f}%", delta_color=delta_color)
        with p_col4:
            st.metric("可用现金", f"${cash:,.0f}")
        
        st.divider()
        
        if positions:
            st.markdown(f"**当前持有 {len(positions)} 只股票**")
            
            # 持仓列表（带详情展示）
            for pos in positions:
                symbol = pos.get('symbol', '')
                shares = pos.get('shares', 0)
                avg_cost = pos.get('avg_cost', 0)
                current_price = pos.get('current_price', avg_cost)
                
                pnl = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
                market_value = shares * current_price
                
                # 颜色
                pnl_color = "#00C853" if pnl >= 0 else "#FF1744"
                
                with st.container():
                    # 持仓卡片
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                border-left: 4px solid {pnl_color};
                                padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.4em; font-weight: bold;">{symbol}</span>
                                <span style="margin-left: 12px; color: #888;">{shares}股</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.2em; color: {pnl_color};">{pnl:+.1f}%</div>
                                <div style="color: #888; font-size: 0.9em;">${market_value:,.0f}</div>
                            </div>
                        </div>
                        <div style="margin-top: 8px; display: flex; gap: 16px; color: #888; font-size: 0.9em;">
                            <span>成本 ${avg_cost:.2f}</span>
                            <span>现价 ${current_price:.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 操作按钮
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    with btn_col1:
                        if st.button(f"📊 分析", key=f"pos_detail_{symbol}", use_container_width=True):
                            st.session_state['portfolio_selected'] = symbol
                    with btn_col2:
                        if st.button(f"➕ 加仓", key=f"pos_add_{symbol}", use_container_width=True):
                            st.session_state['portfolio_add'] = symbol
                    with btn_col3:
                        sell_label = "🔴 止损" if pnl < -5 else ("✅ 止盈" if pnl > 10 else "📤 卖出")
                        if st.button(sell_label, key=f"pos_sell_{symbol}", use_container_width=True):
                            st.session_state['portfolio_sell'] = symbol
            
            # 显示选中持仓的详情
            if st.session_state.get('portfolio_selected'):
                st.divider()
                symbol = st.session_state['portfolio_selected']
                
                header_col1, header_col2 = st.columns([6, 1])
                with header_col1:
                    st.markdown(f"### 📊 {symbol} 持仓分析")
                with header_col2:
                    if st.button("❌ 关闭", key="close_portfolio_detail"):
                        st.session_state['portfolio_selected'] = None
                        st.rerun()
                
                render_unified_stock_detail(
                    symbol=symbol,
                    market=market,
                    key_prefix=f"tab4_portfolio_{symbol}"
                )
            
            # 处理卖出
            if st.session_state.get('portfolio_sell'):
                symbol = st.session_state['portfolio_sell']
                pos = next((p for p in positions if p.get('symbol') == symbol), {})
                
                st.divider()
                st.markdown(f"### 📤 卖出 {symbol}")
                
                with st.form(f"sell_form_{symbol}"):
                    max_shares = pos.get('shares', 0)
                    sell_shares = st.number_input("卖出股数", min_value=1, max_value=max_shares, value=max_shares)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("✅ 确认卖出", type="primary"):
                            try:
                                from services.portfolio_service import paper_sell
                                price = pos.get('current_price', 0)
                                result = paper_sell(symbol, sell_shares, price)
                                if result.get('success'):
                                    pnl = result.get('realized_pnl', 0)
                                    st.success(f"🎉 卖出成功! 盈亏: ${pnl:+.2f}")
                                    st.session_state['portfolio_sell'] = None
                                    st.rerun()
                                else:
                                    st.error(result.get('error', '卖出失败'))
                            except Exception as e:
                                st.error(f"卖出失败: {e}")
                    with col2:
                        if st.form_submit_button("❌ 取消"):
                            st.session_state['portfolio_sell'] = None
                            st.rerun()
        else:
            st.info("📭 暂无持仓")
            st.markdown("前往「发现新股」或「策略精选」寻找买入机会！")
    
    # === 买卖信号 (整合进 Tab1 的 expander) ===
    with work_tab1:
      with st.expander("📊 更多买卖信号详情", expanded=False):
        st.subheader("📊 今日买卖信号")
        st.caption("基于多策略分析的买入/卖出建议")
        
        try:
            from strategies.signal_system import get_signal_manager
            
            signal_manager = get_signal_manager()
            
            # 市场选择
            sig_market_choice = st.radio("市场", ["🇺🇸 美股", "🇨🇳 A股"], horizontal=True, key="daily_sig_tab_market")
            sig_market = "US" if "美股" in sig_market_choice else "CN"
            
            col_refresh, col_filter = st.columns([1, 3])
            with col_refresh:
                if st.button("🔄 刷新信号", key="refresh_signals_tab"):
                    with st.spinner("生成交易信号..."):
                        result = signal_manager.generate_daily_signals(market=sig_market)
                        if 'error' not in result:
                            st.success(f"✅ {result.get('buy_signals', 0)}买入, {result.get('sell_signals', 0)}卖出")
                        else:
                            st.error(result['error'])
            
            with col_filter:
                min_confidence = st.slider("最低信心度", 30, 90, 50, key="sig_tab_confidence")
            
            # 获取并显示信号
            signals = signal_manager.get_todays_signals(market=sig_market)
            
            if signals:
                signals = [s for s in signals if s.get('confidence', 0) >= min_confidence]
            
            if not signals:
                st.info("👋 暂无今日信号，请点击「刷新信号」按钮生成")
            else:
                buy_signals = [s for s in signals if s['signal_type'] == '买入']
                sell_signals = [s for s in signals if s['signal_type'] != '买入']
                price_sym = "¥" if sig_market == "CN" else "$"
                
                # ==========================================
                # 批量交易控制面板
                # ==========================================
                with st.expander("🚀 批量交易", expanded=False):
                    st.markdown("**选择信号 → 选择子账户 → 一键批量买入**")
                    
                    # 获取子账户列表
                    try:
                        from services.portfolio_service import list_paper_accounts, paper_buy
                        batch_accounts = list_paper_accounts()
                        batch_account_names = [a['account_name'] for a in batch_accounts] if batch_accounts else ['default']
                    except Exception:
                        batch_account_names = ['default']
                    
                    batch_col1, batch_col2, batch_col3 = st.columns([2, 1, 1])
                    
                    with batch_col1:
                        # 多选买入信号
                        buy_options = [f"{s['symbol']} ({s['confidence']:.0f}%)" for s in buy_signals]
                        selected_signals = st.multiselect(
                            "选择买入信号",
                            options=buy_options,
                            default=[],
                            key="batch_signal_select",
                            help="按住 Cmd/Ctrl 多选"
                        )
                    
                    with batch_col2:
                        # 目标子账户
                        target_account = st.selectbox(
                            "目标子账户",
                            options=batch_account_names,
                            key="batch_target_account"
                        )
                        
                        # 每只股票投资金额
                        per_stock_amount = st.number_input(
                            "每只投资金额",
                            min_value=500,
                            max_value=50000,
                            value=2000,
                            step=500,
                            key="batch_per_stock_amount"
                        )
                    
                    with batch_col3:
                        st.write("")  # 占位
                        total_cost = len(selected_signals) * per_stock_amount
                        st.markdown(f"""
                        <div style="background: rgba(0,200,83,0.1); padding: 10px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.8em; color: #888;">预估总投资</div>
                            <div style="font-size: 1.3em; font-weight: bold; color: #00C853;">
                                {price_sym}{total_cost:,.0f}
                            </div>
                            <div style="font-size: 0.75em; color: #888;">{len(selected_signals)} 只股票</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 执行批量买入按钮
                    if st.button("🚀 批量模拟买入", type="primary", disabled=len(selected_signals) == 0, key="batch_buy_btn"):
                        if selected_signals:
                            success_count = 0
                            fail_results = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, sig_label in enumerate(selected_signals):
                                symbol = sig_label.split(" ")[0]  # 提取股票代码
                                
                                # 找到对应的信号获取价格
                                matching_sig = next((s for s in buy_signals if s['symbol'] == symbol), None)
                                if not matching_sig:
                                    fail_results.append(f"{symbol}: 信号未找到")
                                    continue
                                
                                price = matching_sig['price']
                                shares = max(1, int(per_stock_amount / price))
                                
                                status_text.text(f"正在买入 {symbol}... ({i+1}/{len(selected_signals)})")
                                
                                try:
                                    result = paper_buy(
                                        symbol=symbol,
                                        shares=shares,
                                        price=price,
                                        market=sig_market,
                                        account_name=target_account
                                    )
                                    
                                    if result.get('success'):
                                        success_count += 1
                                    else:
                                        fail_results.append(f"{symbol}: {result.get('error', '未知错误')}")
                                        
                                except Exception as e:
                                    fail_results.append(f"{symbol}: {str(e)[:30]}")
                                
                                progress_bar.progress((i + 1) / len(selected_signals))
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            # 显示结果
                            if success_count > 0:
                                st.success(f"✅ 成功买入 {success_count} 只股票到【{target_account}】子账户")
                            
                            if fail_results:
                                st.warning(f"⚠️ {len(fail_results)} 只失败:")
                                for fr in fail_results[:5]:
                                    st.caption(f"  • {fr}")
                        else:
                            st.warning("请先选择要买入的信号")
                
                # ==========================================
                # 买卖信号并排显示
                # ==========================================
                col_buy, col_sell = st.columns(2)
                
                with col_buy:
                    st.markdown(f"### 🟢 买入 ({len(buy_signals)})")
                    for sig in buy_signals[:8]:
                        strength_icon = "🔥" if sig['strength'] == '强烈' else "⚡" if sig['strength'] == '中等' else "💧"
                        confidence_color = "#00C853" if sig['confidence'] >= 70 else "#FFD600" if sig['confidence'] >= 50 else "#888"
                        
                        st.markdown(f"""
                        <div style="background: rgba(0,200,83,0.1); border-left: 3px solid #00C853; 
                                    padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: bold;">{sig['symbol']}</span>
                                <span style="color: {confidence_color};">{sig['confidence']:.0f}% {strength_icon}</span>
                            </div>
                            <div style="font-size: 0.85em; color: #888;">{price_sym}{sig['price']:.2f} | {sig['strategy']}</div>
                            <div style="font-size: 0.8em; color: #aaa; margin-top: 4px;">💡 {sig['reason'][:50]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_sell:
                    st.markdown(f"### 🔴 卖出 ({len(sell_signals)})")
                    for sig in sell_signals[:8]:
                        signal_icon = {'止损': '🛑', '止盈': '🎯', '卖出': '🔴'}.get(sig['signal_type'], '📊')
                        
                        st.markdown(f"""
                        <div style="background: rgba(255,82,82,0.1); border-left: 3px solid #FF5252; 
                                    padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: bold;">{sig['symbol']}</span>
                                <span style="color: #FF5252;">{signal_icon} {sig['signal_type']}</span>
                            </div>
                            <div style="font-size: 0.85em; color: #888;">{price_sym}{sig['price']:.2f}</div>
                            <div style="font-size: 0.8em; color: #aaa; margin-top: 4px;">⚠️ {sig['reason'][:50]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 统计
                st.divider()
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("买入信号", len(buy_signals))
                with s2:
                    st.metric("卖出信号", len(sell_signals))
                with s3:
                    avg_conf = sum([s['confidence'] for s in buy_signals]) / len(buy_signals) if buy_signals else 0
                    st.metric("平均信心度", f"{avg_conf:.0f}%")
                with s4:
                    strong = len([s for s in buy_signals if s['strength'] == '强烈'])
                    st.metric("强烈信号", strong)
                    
        except ImportError as e:
            st.warning(f"信号系统模块未加载: {e}")
        except Exception as e:
            st.error(f"信号加载失败: {e}")

    # === Tab 5: 热点板块 (主题篮子 + 龙头追踪 + 社交热度) ===
    with work_tab5:
        st.subheader("🔥 热点板块雷达")
        st.caption("先看主题强弱，再盯龙头，支持一键加入观察列表")

        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            top_themes = st.slider("主题数量", min_value=3, max_value=10, value=5, key=f"theme_top_{market}")
        with ctrl2:
            leaders_per_theme = st.slider("每个主题龙头数", min_value=2, max_value=6, value=4, key=f"theme_leaders_{market}")
        with ctrl3:
            include_social = st.checkbox("叠加社交热度 (Reddit/X)", value=False, key=f"theme_social_{market}")

        with st.expander("🛠️ 网络连通性自检", expanded=False):
            st.caption("用于排查：为何拿不到 Polygon 行情或社交帖子")
            if st.button("🔍 运行自检", key=f"theme_net_diag_{market}"):
                with st.spinner("检测网络连通性..."):
                    diag_df = _run_network_diagnostics()
                    st.dataframe(diag_df, use_container_width=True, hide_index=True)

        radar_state_key = f"theme_radar_cache_{market}"
        trigger_refresh = st.button("🔄 刷新主题雷达", key=f"refresh_theme_radar_{market}")
        if trigger_refresh:
            try:
                _cached_theme_radar.clear()
            except Exception:
                pass

        if trigger_refresh or radar_state_key not in st.session_state:
            with st.spinner("计算主题热度与龙头强度..."):
                try:
                    st.session_state[radar_state_key] = _cached_theme_radar(
                        market=market,
                        top_themes=top_themes,
                        leaders_per_theme=leaders_per_theme,
                        include_social=include_social,
                        latest_date=latest_date,
                    )
                except Exception as e:
                    st.error(f"主题雷达加载失败: {e}")
                    st.session_state[radar_state_key] = None

        radar = st.session_state.get(radar_state_key)
        if not radar or not radar.get("themes"):
            st.info("暂无可用主题数据，请点击刷新或检查行情数据源。")
            try:
                from data_fetcher import get_recent_fetch_errors
                recent_errs = get_recent_fetch_errors(limit=10)
                if recent_errs:
                    st.markdown("#### 🧪 数据源诊断（最近错误）")
                    st.dataframe(pd.DataFrame(recent_errs), use_container_width=True, hide_index=True)
            except Exception:
                pass
        else:
            themes = radar.get("themes", [])
            radar_meta = radar.get("meta", {})
            social_meta = radar_meta.get("social", {})
            if radar.get("errors"):
                st.caption(f"⚠️ 数据抓取异常 {len(radar['errors'])} 条（已自动跳过异常股票）")
                with st.expander("查看异常明细", expanded=False):
                    err_df = pd.DataFrame({"error": radar.get("errors", [])[:30]})
                    st.dataframe(err_df, use_container_width=True, hide_index=True)

            # 社交热度状态看板
            if include_social:
                reason = social_meta.get("reason", "unknown")
                if social_meta.get("enabled"):
                    st.success("✅ 社交热度已启用：数据来自 Reddit/X (ddgs 搜索聚合)")
                elif reason == "missing_or_broken_ddgs":
                    st.warning("⚠️ 社交热度未生效：`ddgs` 不可用（未安装或导入失败）")
                    if social_meta.get("error"):
                        st.caption(f"异常信息: {social_meta.get('error')}")
                elif reason == "source_blocked":
                    st.warning("⚠️ 社交热度受限：网络可达，但被源站拦截或限流（常见于云端IP）")
                    if social_meta.get("error"):
                        st.caption(f"异常信息: {social_meta.get('error')}")
                else:
                    st.warning("⚠️ 社交热度未生效：运行时异常，已回退到纯行情模式")
                    if social_meta.get("error"):
                        st.caption(f"异常信息: {social_meta.get('error')}")
                    else:
                        social_errs = [e for e in (radar.get("errors", []) or []) if str(e).startswith("social")]
                        if social_errs:
                            st.caption(f"异常信息: {social_errs[0]}")
            else:
                st.caption("社交热度当前未开启。勾选上方「叠加社交热度 (Reddit/X)」可启用。")

            # 主题总览
            summary_rows = []
            for idx, t in enumerate(themes, start=1):
                social = t.get("social")
                summary_rows.append({
                    "排名": idx,
                    "主题": t.get("theme", "-"),
                    "热度分": t.get("theme_score", 0),
                    "20日均涨幅": f"{t.get('avg_ret20', 0):+.1f}%",
                    "60日均涨幅": f"{t.get('avg_ret60', 0):+.1f}%",
                    "正收益占比": f"{t.get('positive_ratio', 0):.0f}%",
                    "社交情绪": f"{social.get('avg_sentiment', 0):+.2f}" if social else "-",
                })

            st.markdown("### 🧭 主题强度排行")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            if include_social:
                social_rows = []
                for idx, t in enumerate(themes, start=1):
                    social = t.get("social")
                    if not social:
                        continue
                    leaders = [x.get("symbol", "") for x in t.get("leaders", [])[:2]]
                    social_rows.append({
                        "排名": idx,
                        "主题": t.get("theme", "-"),
                        "跟踪龙头": ", ".join([s for s in leaders if s]),
                        "帖子数": social.get("total_posts", 0),
                        "Bull": social.get("bullish_count", 0),
                        "Bear": social.get("bearish_count", 0),
                        "情绪分": f"{social.get('avg_sentiment', 0):+.2f}",
                    })
                if social_rows:
                    st.markdown("### 📣 社交热度榜")
                    st.dataframe(pd.DataFrame(social_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("社交热度已开启，但当前主题暂无可用帖子样本。")

            # 龙头明细
            st.markdown("### 🏆 主题龙头明细")
            for idx, theme in enumerate(themes, start=1):
                leaders = theme.get("leaders", [])
                if not leaders:
                    continue

                social = theme.get("social")
                social_text = ""
                if social:
                    social_text = (
                        f" | 社交: 情绪 {social.get('avg_sentiment', 0):+.2f}"
                        f" (Bull {social.get('bullish_count', 0)} / Bear {social.get('bearish_count', 0)})"
                    )

                with st.expander(
                    f"#{idx} {theme.get('theme', '-')}"
                    f" | 热度 {theme.get('theme_score', 0):.1f}"
                    f"{social_text}",
                    expanded=(idx == 1),
                ):
                    leader_df = pd.DataFrame([{
                        "代码": x.get("symbol", ""),
                        "现价": f"{x.get('price', 0):.2f}",
                        "20日涨幅": f"{x.get('ret20', 0):+.1f}%",
                        "60日涨幅": f"{x.get('ret60', 0):+.1f}%",
                        "量比(5/20)": f"{x.get('vol_ratio', 0):.2f}" if pd.notna(x.get("vol_ratio")) else "-",
                        "BLUE(日)": f"{x.get('blue_daily', 0):.0f}" if pd.notna(x.get("blue_daily")) else "-",
                        "龙头分": f"{x.get('leader_score', 0):.1f}",
                    } for x in leaders])

                    st.dataframe(leader_df, use_container_width=True, hide_index=True)

                    a1, a2 = st.columns([2, 1])
                    with a1:
                        st.caption("可直接把该主题Top龙头加入观察列表，做每日跟踪。")
                    with a2:
                        if st.button(f"⭐ 关注Top{min(3, len(leaders))}", key=f"watch_theme_{market}_{idx}"):
                            try:
                                from services.theme_radar_service import add_theme_leaders_to_watchlist

                                ret = add_theme_leaders_to_watchlist(
                                    theme_data=theme,
                                    market=market,
                                    top_n=min(3, len(leaders)),
                                )
                                if ret.get("added", 0) > 0:
                                    st.success(f"✅ 已加入 {ret['added']} 只：{ret.get('theme', '')}")
                                if ret.get("failed"):
                                    st.warning(f"⚠️ 加入失败: {', '.join(ret['failed'])}")
                            except Exception as e:
                                st.error(f"加入观察失败: {e}")

    # ============================================
    # 📌 SOP Step 3: 收盘复盘（组合追踪，在 Tabs 下方独立区域）
    # ============================================
    st.divider()
    with st.expander("📌 ③ 收盘复盘 — 信号组合追踪", expanded=False):
        st.subheader("📌 信号组合持续追踪")
        st.caption("自动追踪：日/周/月BLUE、日/周/月黑马与筹码标签的各种组合表现")

        # 组合层 + 极致提升（Tab内版本）
        st.markdown("### 🧩 策略组合层（Meta Allocator）")
        if tracking_rows_for_action:
            tab7_r1, tab7_r2, tab7_r3, tab7_r4 = st.columns(4)
            with tab7_r1:
                tab7_exit_rule = st.selectbox(
                    "平仓规则",
                    options=["fixed_5d", "fixed_10d", "fixed_20d", "tp_sl_time", "kdj_dead_cross", "top_divergence_guard", "duokongwang_sell"],
                    index=1,
                    key=f"track_exit_rule_tab7_{market}",
                )
            with tab7_r2:
                tab7_rule_tp = st.slider("止盈(%)", min_value=3, max_value=25, value=10, step=1, key=f"track_rule_tp_tab7_{market}")
            with tab7_r3:
                tab7_rule_sl = st.slider("止损(%)", min_value=2, max_value=20, value=6, step=1, key=f"track_rule_sl_tab7_{market}")
            with tab7_r4:
                tab7_rule_max_hold = st.slider("最长持有天", min_value=5, max_value=60, value=20, step=1, key=f"track_rule_hold_tab7_{market}")

            tab7_a1, tab7_a2, tab7_a3, tab7_a4 = st.columns(4)
            with tab7_a1:
                tab7_alloc_fee_bps = st.slider("手续费(bps)", min_value=0.0, max_value=30.0, value=5.0, step=0.5, key=f"track_alloc_fee_bps_tab7_{market}")
            with tab7_a2:
                tab7_alloc_slip_bps = st.slider("滑点(bps)", min_value=0.0, max_value=30.0, value=5.0, step=0.5, key=f"track_alloc_slip_bps_tab7_{market}")
            with tab7_a3:
                tab7_alloc_min_samples = st.slider("最小样本", min_value=8, max_value=120, value=20, step=2, key=f"track_alloc_min_samples_tab7_{market}")
            with tab7_a4:
                tab7_alloc_top_n = st.slider("当日候选数", min_value=5, max_value=30, value=12, step=1, key=f"track_alloc_topn_tab7_{market}")

            tab7_auto_best_exit = st.checkbox(
                "每个策略自动选择最优卖出规则",
                value=True,
                key=f"track_alloc_auto_best_exit_tab7_{market}",
            )

            if tab7_auto_best_exit:
                tab7_exit_rule_candidates = [
                    {"rule_name": "fixed_5d", "take_profit_pct": float(tab7_rule_tp), "stop_loss_pct": float(tab7_rule_sl), "max_hold_days": int(tab7_rule_max_hold)},
                    {"rule_name": "fixed_10d", "take_profit_pct": float(tab7_rule_tp), "stop_loss_pct": float(tab7_rule_sl), "max_hold_days": int(tab7_rule_max_hold)},
                    {"rule_name": "fixed_20d", "take_profit_pct": float(tab7_rule_tp), "stop_loss_pct": float(tab7_rule_sl), "max_hold_days": int(tab7_rule_max_hold)},
                    {"rule_name": "tp_sl_time", "take_profit_pct": float(tab7_rule_tp), "stop_loss_pct": float(tab7_rule_sl), "max_hold_days": int(tab7_rule_max_hold)},
                    {"rule_name": "kdj_dead_cross", "take_profit_pct": float(tab7_rule_tp), "stop_loss_pct": float(tab7_rule_sl), "max_hold_days": int(tab7_rule_max_hold)},
                    {"rule_name": "top_divergence_guard", "take_profit_pct": float(tab7_rule_tp), "stop_loss_pct": float(tab7_rule_sl), "max_hold_days": int(tab7_rule_max_hold)},
                    {"rule_name": "duokongwang_sell", "take_profit_pct": float(tab7_rule_tp), "stop_loss_pct": float(tab7_rule_sl), "max_hold_days": int(tab7_rule_max_hold)},
                ]
                tab7_perf_rows = evaluate_strategy_baskets_best_exit(
                    rows=tracking_rows_for_action,
                    exit_rule_candidates=tab7_exit_rule_candidates,
                    fee_bps=float(tab7_alloc_fee_bps),
                    slippage_bps=float(tab7_alloc_slip_bps),
                    min_samples=int(tab7_alloc_min_samples),
                    max_rows=20000,
                )
                st.caption("当前按“每个策略最优卖出规则”排序与分配权重。")
            else:
                tab7_perf_rows = evaluate_strategy_baskets(
                    rows=tracking_rows_for_action,
                    rule_name=tab7_exit_rule,
                    take_profit_pct=float(tab7_rule_tp),
                    stop_loss_pct=float(tab7_rule_sl),
                    max_hold_days=int(tab7_rule_max_hold),
                    fee_bps=float(tab7_alloc_fee_bps),
                    slippage_bps=float(tab7_alloc_slip_bps),
                    min_samples=int(tab7_alloc_min_samples),
                    max_rows=20000,
                )

            tab7_perf_df = pd.DataFrame(tab7_perf_rows) if tab7_perf_rows else pd.DataFrame()
            if not tab7_perf_df.empty:
                tab7_show_cols = [
                    "策略", "sample", "net_win_rate_pct", "net_avg_return_pct",
                    "total_return_pct", "ann_return_pct", "ann_return_raw_pct", "ann_return_expect_pct",
                    "sharpe", "max_drawdown_pct",
                    "profit_factor", "turnover_per_year", "meta_score",
                ]
                if "exit_rule_desc" in tab7_perf_df.columns:
                    tab7_show_cols.extend(["exit_rule", "exit_rule_desc"])
                tab7_col_map = {
                    "sample": "样本",
                    "net_win_rate_pct": "净胜率(%)",
                    "net_avg_return_pct": "单笔净均收(%)",
                    "total_return_pct": "总收益(%)",
                    "ann_return_pct": "几何年化(主,%)",
                    "ann_return_raw_pct": "原始几何年化(%)",
                    "ann_return_expect_pct": "期望年化(均值,%)",
                    "sharpe": "Sharpe",
                    "max_drawdown_pct": "最大回撤(%)",
                    "profit_factor": "盈亏比",
                    "turnover_per_year": "年换手(次)",
                    "meta_score": "组合评分",
                    "exit_rule": "最优卖出规则",
                    "exit_rule_desc": "规则参数",
                }
                st.dataframe(
                    tab7_perf_df[tab7_show_cols].rename(columns=tab7_col_map),
                    use_container_width=True,
                    hide_index=True,
                )

                tab7_weight_rows = allocate_meta_weights(tab7_perf_rows, max_weight=0.45, min_weight=0.05)
                tab7_weight_df = pd.DataFrame(tab7_weight_rows) if tab7_weight_rows else pd.DataFrame()
                tab7_b1, tab7_b2 = st.columns([1, 1])
                with tab7_b1:
                    st.markdown("**动态权重建议**")
                    if not tab7_weight_df.empty:
                        st.dataframe(tab7_weight_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("暂无可分配策略")
                with tab7_b2:
                    st.markdown("**当日组合候选（按权重）**")
                    tab7_candidate_mode = st.radio(
                        "候选池口径",
                        options=["仅当日新信号", "含历史延续信号"],
                        horizontal=True,
                        key=f"track_alloc_candidate_mode_tab7_{market}",
                    )
                    tab7_exec_capital = st.number_input(
                        "执行资金($)",
                        min_value=1000.0,
                        max_value=5000000.0,
                        value=100000.0,
                        step=1000.0,
                        key=f"track_alloc_exec_capital_tab7_{market}",
                    )
                    tab7_history_days = 20
                    if tab7_candidate_mode == "含历史延续信号":
                        tab7_history_days = st.slider(
                            "历史延续窗口(天)",
                            min_value=3,
                            max_value=60,
                            value=20,
                            step=1,
                            key=f"track_alloc_history_days_tab7_{market}",
                        )
                    try:
                        tab7_today_plan = build_today_meta_plan(
                            rows=tracking_rows_for_action,
                            weight_rows=tab7_weight_rows,
                            top_n=int(tab7_alloc_top_n),
                            total_capital=float(tab7_exec_capital),
                            include_history=(tab7_candidate_mode == "含历史延续信号"),
                            max_signal_age_days=int(tab7_history_days),
                        )
                    except TypeError:
                        try:
                            tab7_today_plan = build_today_meta_plan(
                                rows=tracking_rows_for_action,
                                weight_rows=tab7_weight_rows,
                                top_n=int(tab7_alloc_top_n),
                                total_capital=float(tab7_exec_capital),
                            )
                        except TypeError:
                            tab7_today_plan = build_today_meta_plan(
                                rows=tracking_rows_for_action,
                                weight_rows=tab7_weight_rows,
                                top_n=int(tab7_alloc_top_n),
                            )
                    tab7_today_plan_df = pd.DataFrame(tab7_today_plan) if tab7_today_plan else pd.DataFrame()
                    if not tab7_today_plan_df.empty:
                        if tab7_candidate_mode == "仅当日新信号":
                            tab7_show_cols_plan = [
                                "信号日期",
                                "symbol",
                                "主策略",
                                "命中策略数",
                                "命中策略",
                                "策略权重合计(%)",
                                "综合执行分(0-100)",
                                "建议仓位(%)",
                                "建议金额($)",
                                "买入触发价",
                                "失效价",
                                "R值",
                                "第一止盈价(2R)",
                                "执行状态",
                                "blue_daily",
                                "blue_weekly",
                            ]
                        else:
                            tab7_show_cols_plan = [
                                "信号日期",
                                "symbol",
                                "主策略",
                                "距信号天数",
                                "命中策略数",
                                "命中策略",
                                "策略权重合计(%)",
                                "综合执行分(0-100)",
                                "建议仓位(%)",
                                "建议金额($)",
                                "信号价",
                                "现价",
                                "价格变化($)",
                                "价格变化(%)",
                                "买入触发价",
                                "失效价",
                                "R值",
                                "第一止盈价(2R)",
                                "执行状态",
                                "blue_daily",
                                "blue_weekly",
                            ]
                        tab7_today_plan_df = tab7_today_plan_df[[c for c in tab7_show_cols_plan if c in tab7_today_plan_df.columns]]
                        st.dataframe(tab7_today_plan_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("今日暂无满足组合规则的候选。")
            else:
                st.info("组合层样本不足：请先积累更多候选追踪样本。")

            st.markdown("### 🔬 极致条件提升（Lift）")
            tab7_lift_ret = _analyze_extreme_lift(
                market=market,
                days_back=360,
                exit_rule=tab7_exit_rule,
                take_profit_pct=float(tab7_rule_tp),
                stop_loss_pct=float(tab7_rule_sl),
                max_hold_days=int(tab7_rule_max_hold),
                max_rows=20000,
                schema_ver=2,
            )
            if tab7_lift_ret.get("ok"):
                tab7_lift_df = pd.DataFrame(tab7_lift_ret.get("table") or [])
                if not tab7_lift_df.empty:
                    st.dataframe(tab7_lift_df, use_container_width=True, hide_index=True)
                else:
                    st.info("暂无可用组合统计")
            else:
                st.info("极致条件统计暂无样本，请先执行回填与刷新。")
        else:
            st.info("暂无候选追踪样本，先运行扫描并回填历史。")

        st.divider()

        combo_templates = {
            "不使用模板": {"required_all": [], "required_any_groups": [], "keyword": ""},
            "日BLUE 基础线": {"required_all": ["DAY_BLUE"], "required_any_groups": [], "keyword": "DAY_BLUE"},
            "周BLUE 趋势线": {"required_all": ["WEEK_BLUE"], "required_any_groups": [], "keyword": "WEEK_BLUE"},
            "月BLUE 中长线": {"required_all": ["MONTH_BLUE"], "required_any_groups": [], "keyword": "MONTH_BLUE"},
            "日+周 共振": {"required_all": ["DAY_BLUE", "WEEK_BLUE"], "required_any_groups": [], "keyword": "DAY_BLUE+WEEK_BLUE"},
            "日+月 共振": {"required_all": ["DAY_BLUE", "MONTH_BLUE"], "required_any_groups": [], "keyword": "DAY_BLUE+MONTH_BLUE"},
            "日周月 三线共振": {"required_all": ["DAY_BLUE", "WEEK_BLUE", "MONTH_BLUE"], "required_any_groups": [], "keyword": "DAY_BLUE+MONTH_BLUE+WEEK_BLUE"},
            "黑马(日/周/月 任一)+日BLUE": {
                "required_all": ["DAY_BLUE"],
                "required_any_groups": [["DAY_HEIMA", "WEEK_HEIMA", "MONTH_HEIMA"]],
                "keyword": "",
            },
            "筹码突破 + 日周共振": {
                "required_all": ["DAY_BLUE", "WEEK_BLUE", "CHIP_BREAKOUT"],
                "required_any_groups": [],
                "keyword": "",
            },
            "中长线强共振(三线+黑马任一+筹码密集)": {
                "required_all": ["DAY_BLUE", "WEEK_BLUE", "MONTH_BLUE", "CHIP_DENSE"],
                "required_any_groups": [["DAY_HEIMA", "WEEK_HEIMA", "MONTH_HEIMA"]],
                "keyword": "",
            },
        }

        t1, t2 = st.columns([2, 2])
        with t1:
            selected_template_name = st.selectbox(
                "组合模板",
                options=list(combo_templates.keys()),
                index=0,
                key=f"track_template_{market}",
            )
        with t2:
            use_template = st.checkbox(
                "应用模板过滤",
                value=(selected_template_name != "不使用模板"),
                key=f"track_template_enable_{market}",
            )

        pin_key = f"track_pinned_templates_{market}"
        pin_db_key = f"{pin_key}_v1"
        pin_default = ["日周月 三线共振", "中长线强共振(三线+黑马任一+筹码密集)"]
        available_pin_options = [x for x in combo_templates.keys() if x != "不使用模板"]

        if pin_key not in st.session_state:
            restored_pins = list(pin_default)
            try:
                saved_pin_raw = get_app_setting(pin_db_key, default=None)
                if saved_pin_raw:
                    parsed = json.loads(saved_pin_raw)
                    if isinstance(parsed, list):
                        restored_pins = [
                            x for x in parsed
                            if x in available_pin_options
                        ]
            except Exception as e:
                print(f"⚠️ 读取置顶模板失败({pin_db_key}): {e}")
            if not restored_pins:
                restored_pins = list(pin_default)
            st.session_state[pin_key] = restored_pins
        p1, p2 = st.columns([3, 1])
        with p1:
            pinned_templates = st.multiselect(
                "⭐ 置顶模板（用于组合对比/推送）",
                options=available_pin_options,
                default=[x for x in st.session_state[pin_key] if x in combo_templates and x != "不使用模板"],
                key=f"track_pin_select_{market}",
            )
            old_pins = list(st.session_state.get(pin_key, []))
            st.session_state[pin_key] = pinned_templates
            if pinned_templates != old_pins:
                try:
                    set_app_setting(pin_db_key, json.dumps(pinned_templates, ensure_ascii=False))
                except Exception as e:
                    st.caption(f"⚠️ 置顶模板保存失败: {e}")
        with p2:
            if st.button("使用置顶模板", key=f"track_use_pinned_{market}"):
                if pinned_templates:
                    selected_template_name = pinned_templates[0]
                    use_template = True
                    st.success(f"已切换到置顶模板: {selected_template_name}")
                else:
                    st.info("请先选择置顶模板")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            days_back = st.slider("回看天数", min_value=30, max_value=720, value=180, step=30, key=f"track_days_{market}")
        with c2:
            min_samples = st.slider("最小样本", min_value=3, max_value=50, value=8, step=1, key=f"track_min_samples_{market}")
        with c3:
            if st.button("🔄 刷新追踪数据", key=f"track_refresh_btn_{market}"):
                with st.spinner("刷新候选追踪中..."):
                    refreshed = refresh_candidate_tracking(market=market, max_rows=2000)
                st.success(f"已刷新 {refreshed} 条记录")
        with c4:
            st.caption(f"市场: {market}")
            if st.button("📥 回填历史扫描", key=f"track_backfill_btn_{market}"):
                with st.spinner("回填历史扫描信号中..."):
                    added = backfill_candidates_from_scan_history(market=market, recent_days=9999, max_per_day=800)
                    refreshed = refresh_candidate_tracking(market=market, max_rows=5000)
                st.success(f"回填完成: {added} 条 | 刷新: {refreshed} 条")

        with st.expander("⚙️ 标签规则（黑马/筹码/蓝线）", expanded=False):
            r1, r2, r3 = st.columns(3)
            with r1:
                day_blue_min = st.slider(
                    "DAY_BLUE 阈值", min_value=50, max_value=180,
                    value=int(DEFAULT_TAG_RULES["day_blue_min"]), step=5, key=f"rule_day_blue_{market}"
                )
                week_blue_min = st.slider(
                    "WEEK_BLUE 阈值", min_value=40, max_value=160,
                    value=int(DEFAULT_TAG_RULES["week_blue_min"]), step=5, key=f"rule_week_blue_{market}"
                )
            with r2:
                month_blue_min = st.slider(
                    "MONTH_BLUE 阈值", min_value=30, max_value=140,
                    value=int(DEFAULT_TAG_RULES["month_blue_min"]), step=5, key=f"rule_month_blue_{market}"
                )
                chip_dense_min = st.slider(
                    "CHIP_DENSE 最低获利盘", min_value=0.40, max_value=0.95,
                    value=float(DEFAULT_TAG_RULES["chip_dense_profit_ratio_min"]), step=0.05, key=f"rule_chip_dense_{market}"
                )
            with r3:
                chip_breakout_min = st.slider(
                    "CHIP_BREAKOUT 最低获利盘", min_value=0.60, max_value=0.99,
                    value=float(DEFAULT_TAG_RULES["chip_breakout_profit_ratio_min"]), step=0.01, key=f"rule_chip_break_{market}"
                )
                chip_overhang_max = st.slider(
                    "CHIP_OVERHANG 最高获利盘", min_value=0.05, max_value=0.50,
                    value=float(DEFAULT_TAG_RULES["chip_overhang_profit_ratio_max"]), step=0.01, key=f"rule_chip_overhang_{market}"
                )

            rule_cfg = {
                "day_blue_min": float(day_blue_min),
                "week_blue_min": float(week_blue_min),
                "month_blue_min": float(month_blue_min),
                "chip_dense_profit_ratio_min": float(chip_dense_min),
                "chip_breakout_profit_ratio_min": float(chip_breakout_min),
                "chip_overhang_profit_ratio_max": float(chip_overhang_max),
            }
            if st.button("🧠 重新计算历史标签", key=f"rule_reclassify_{market}"):
                with st.spinner("重算历史标签中..."):
                    changed = reclassify_tracking_tags(market=market, rules=rule_cfg, max_rows=5000)
                st.success(f"已重算 {changed} 条历史标签")

        rows = get_candidate_tracking_rows(market=market, days_back=days_back)
        if not rows:
            st.info("暂无追踪数据。请先运行每日扫描并进入“每日工作台”。")
        else:
            all_rows = list(rows)
            f1, f2 = st.columns([2, 3])
            with f1:
                required_tags = st.multiselect(
                    "按标签过滤（必须全部包含）",
                    options=CORE_TAGS,
                    default=[],
                    key=f"track_tag_filter_{market}",
                )
            with f2:
                combo_contains = st.text_input(
                    "组合关键词过滤（例如 DAY_BLUE+WEEK_BLUE）",
                    value="",
                    key=f"track_combo_filter_{market}",
                ).strip().upper()

            template_cfg = combo_templates.get(selected_template_name, combo_templates["不使用模板"])
            effective_required = list(required_tags)
            effective_keyword = combo_contains
            required_any_groups = []

            if use_template:
                for tag in template_cfg.get("required_all", []):
                    if tag not in effective_required:
                        effective_required.append(tag)
                required_any_groups = template_cfg.get("required_any_groups", []) or []
                if not effective_keyword:
                    effective_keyword = str(template_cfg.get("keyword", "") or "").upper()

            if effective_required:
                rows = [
                    r for r in rows
                    if set(effective_required).issubset(set(r.get("signal_tags_list") or []))
                ]

            for any_group in required_any_groups:
                rows = [
                    r for r in rows
                    if any(tag in set(r.get("signal_tags_list") or []) for tag in any_group)
                ]

            # 顶部总览
            total = len(rows)
            combo_stats = []
            if total == 0:
                st.info("当前过滤条件下无样本。请放宽过滤条件或关闭模板过滤。")
            else:
                wins = sum(1 for r in rows if float(r.get("pnl_pct") or 0) > 0)
                avg_pnl = np.mean([float(r.get("pnl_pct") or 0) for r in rows]) if rows else 0
                d2p_vals = [int(r["first_positive_day"]) for r in rows if r.get("first_positive_day") is not None]
                median_d2p = int(np.median(d2p_vals)) if d2p_vals else None

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("追踪样本", total)
                m2.metric("当前胜率", f"{(wins / total * 100):.1f}%")
                m3.metric("当前平均收益", f"{avg_pnl:+.2f}%")
                m4.metric("首次转正中位天数", f"{median_d2p}天" if median_d2p is not None else "-")

                st.markdown("### 🧩 组合绩效矩阵")
                combo_stats = build_combo_stats(rows, min_samples=min_samples)
            if combo_stats:
                if effective_keyword:
                    combo_stats = [x for x in combo_stats if effective_keyword in str(x.get("组合", "")).upper()]
                combo_df = pd.DataFrame(combo_stats)
                st.dataframe(combo_df, use_container_width=True, hide_index=True)

                # 置顶模板对比图
                st.markdown("### 📈 置顶模板对比")
                selected_compare_templates = [
                    x for x in st.session_state.get(pin_key, []) if x in combo_templates and x != "不使用模板"
                ][:3]
                if selected_compare_templates:
                    def _filter_by_template(source_rows, template_cfg):
                        out = list(source_rows)
                        req_all = template_cfg.get("required_all", []) or []
                        req_any_groups = template_cfg.get("required_any_groups", []) or []
                        keyword = str(template_cfg.get("keyword", "") or "").upper()

                        for tag in req_all:
                            out = [
                                r for r in out
                                if tag in set(r.get("signal_tags_list") or [])
                            ]
                        for any_group in req_any_groups:
                            out = [
                                r for r in out
                                if any(tag in set(r.get("signal_tags_list") or []) for tag in any_group)
                            ]
                        if keyword:
                            out = [
                                r for r in out
                                if keyword in str(r.get("signal_tags", "")).upper()
                            ]
                        return out

                    compare_rows = []
                    for tname in selected_compare_templates:
                        tcfg = combo_templates.get(tname, {})
                        matched_rows = _filter_by_template(all_rows, tcfg)
                        if len(matched_rows) < min_samples:
                            continue
                        wins_t = sum(1 for r in matched_rows if float(r.get("pnl_pct") or 0) > 0)
                        avg_t = float(np.mean([float(r.get("pnl_pct") or 0) for r in matched_rows])) if matched_rows else 0.0
                        d2p_vals_t = [
                            int(r["first_positive_day"]) for r in matched_rows
                            if r.get("first_positive_day") is not None
                        ]
                        med_t = int(np.median(d2p_vals_t)) if d2p_vals_t else None
                        combo_label = "+".join(tcfg.get("required_all", []) or []) or "模板规则"
                        compare_rows.append({
                            "模板": tname,
                            "组合": combo_label,
                            "样本数": int(len(matched_rows)),
                            "当前胜率(%)": float(wins_t / len(matched_rows) * 100),
                            "当前平均收益(%)": float(avg_t),
                            "首次转正中位天数": med_t,
                        })
                    if compare_rows:
                        compare_df = pd.DataFrame(compare_rows)
                        st.dataframe(compare_df, use_container_width=True, hide_index=True)

                        fig_cmp = go.Figure()
                        fig_cmp.add_trace(
                            go.Bar(
                                x=compare_df["模板"],
                                y=compare_df["当前胜率(%)"],
                                name="当前胜率(%)",
                                marker_color="#2E7D32",
                            )
                        )
                        fig_cmp.add_trace(
                            go.Scatter(
                                x=compare_df["模板"],
                                y=compare_df["当前平均收益(%)"],
                                mode="lines+markers",
                                name="当前平均收益(%)",
                                yaxis="y2",
                                line=dict(color="#1565C0", width=2),
                            )
                        )
                        fig_cmp.update_layout(
                            height=340,
                            xaxis_title="模板",
                            yaxis=dict(title="胜率(%)"),
                            yaxis2=dict(title="平均收益(%)", overlaying="y", side="right"),
                            margin=dict(l=20, r=20, t=20, b=20),
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)

                        # 一键推送置顶模板摘要
                        if st.button("📣 推送置顶模板表现", key=f"track_push_pinned_{market}"):
                            try:
                                from services.notification import NotificationManager
                                nm = NotificationManager()
                                lines = [
                                    f"*📌 置顶模板表现 | {market}*",
                                    f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                    "",
                                ]
                                for _, r in compare_df.iterrows():
                                    lines.append(
                                        f"- {r['模板']} ({r['组合']}): 胜率 {r['当前胜率(%)']:.1f}% | "
                                        f"均收 {r['当前平均收益(%)']:+.2f}% | 样本 {int(r['样本数'])}"
                                    )
                                lines.append("")
                                lines.append("仅供研究，不构成投资建议。")
                                msg = "\n".join(lines)

                                tg_ok = nm.send_telegram(msg) if nm.telegram_token else False
                                wc_ok = nm.send_wecom(msg, msg_type="markdown") if nm.wecom_webhook else False
                                wx_ok = nm.send_wxpusher(
                                    title=f"Coral Creek 置顶模板表现 {market}",
                                    content=msg,
                                ) if nm.wxpusher_app_token else False
                                bark_ok = nm.send_bark(
                                    title=f"置顶模板表现 {market}",
                                    content=msg,
                                ) if nm.bark_url else False
                                st.success(
                                    f"推送完成 | telegram={tg_ok}, wecom={wc_ok}, wxpusher={wx_ok}, bark={bark_ok}"
                                )
                            except Exception as e:
                                st.error(f"推送失败: {e}")
                    else:
                        st.info("置顶模板当前无可匹配组合数据。")
                else:
                    st.info("请先选择置顶模板（最多取前3个做对比）。")
            else:
                st.info("当前组合样本不足，调低“最小样本”可查看更多组合。")

            st.markdown("### 🧱 分层统计")
            seg1, seg2 = st.columns(2)
            with seg1:
                cap_df = pd.DataFrame(build_segment_stats(rows, by="cap_category"))
                st.markdown("**按市值层**")
                st.dataframe(cap_df, use_container_width=True, hide_index=True)
            with seg2:
                ind_df = pd.DataFrame(build_segment_stats(rows, by="industry"))
                st.markdown("**按板块/行业**")
                st.dataframe(ind_df.head(20), use_container_width=True, hide_index=True)

            st.markdown("### 📋 个股追踪明细")
            detail_df = pd.DataFrame(rows)
            if not detail_df.empty:
                detail_df["标签"] = detail_df["signal_tags_list"].apply(
                    lambda x: ",".join(x[:6]) + ("..." if len(x) > 6 else "") if isinstance(x, list) else ""
                )
                show_cols = [
                    "symbol", "signal_date", "signal_price", "current_price", "pnl_pct",
                    "days_since_signal", "first_positive_day", "max_up_pct", "max_drawdown_pct",
                    "pnl_d1", "pnl_d3", "pnl_d5", "pnl_d10", "pnl_d20",
                    "cap_category", "industry", "标签", "status",
                ]
                show_cols = [c for c in show_cols if c in detail_df.columns]
                show_df = detail_df[show_cols].copy()
                show_df = show_df.rename(
                    columns={
                        "symbol": "代码",
                        "signal_date": "信号日",
                        "signal_price": "信号价",
                        "current_price": "现价",
                        "pnl_pct": "当前收益%",
                        "days_since_signal": "追踪天数",
                        "first_positive_day": "首次转正天",
                        "max_up_pct": "最大浮盈%",
                        "max_drawdown_pct": "最大浮亏%",
                        "cap_category": "市值层",
                        "industry": "行业",
                        "status": "状态",
                    }
                )
                st.dataframe(show_df, use_container_width=True, hide_index=True)

            # 组合 -> 个股钻取
            if combo_stats:
                st.markdown("### 🔎 组合钻取")
                combo_names = [x.get("组合", "") for x in combo_stats if x.get("组合")]
                selected_combo = st.selectbox(
                    "选择一个组合查看个股明细",
                    options=combo_names,
                    key=f"combo_drill_{market}",
                )
                combo_parts = [p for p in str(selected_combo).split("+") if p]
                drill_rows = [
                    r for r in rows
                    if set(combo_parts).issubset(set(r.get("signal_tags_list") or []))
                ]
                if drill_rows:
                    drill_df = pd.DataFrame(drill_rows)
                    drill_df["标签"] = drill_df["signal_tags_list"].apply(
                        lambda x: ",".join(x[:8]) if isinstance(x, list) else ""
                    )
                    drill_show = drill_df.rename(
                        columns={
                            "symbol": "代码",
                            "signal_date": "信号日",
                            "signal_price": "信号价",
                            "current_price": "现价",
                            "pnl_pct": "当前收益%",
                            "days_since_signal": "追踪天数",
                            "first_positive_day": "首次转正天",
                            "cap_category": "市值层",
                            "industry": "行业",
                        }
                    )
                    keep_cols = [
                        "代码", "信号日", "信号价", "现价", "当前收益%",
                        "追踪天数", "首次转正天", "市值层", "行业", "标签",
                    ]
                    keep_cols = [c for c in keep_cols if c in drill_show.columns]
                    st.dataframe(drill_show[keep_cols], use_container_width=True, hide_index=True)

                    symbol_options = [f"{r.get('symbol')} | {r.get('signal_date')}" for r in drill_rows]
                    selected_symbol_row = st.selectbox(
                        "查看个股详细分析",
                        options=symbol_options,
                        key=f"combo_drill_symbol_{market}",
                    )
                    selected_symbol = selected_symbol_row.split("|")[0].strip()
                    if st.button("📊 打开个股详情", key=f"combo_open_detail_{market}"):
                        render_unified_stock_detail(
                            symbol=selected_symbol,
                            market=market,
                            show_charts=True,
                            show_chips=True,
                            show_news=False,
                            show_actions=False,
                            key_prefix=f"combo_drill_{market}_{selected_symbol}",
                        )
                else:
                    st.info("该组合暂无对应个股样本。")

    
    # ============================================
    # 💼 浮动持仓栏 - Alpaca Paper Trading (页面底部)
    # ============================================
    try:
        from components.alpaca_widget import render_alpaca_floating_bar
        render_alpaca_floating_bar(enabled=(market == "US"), market=market)
    except ImportError:
        pass  # 组件未安装时静默跳过


# Legacy code removed - all functionality is now in the 4 redesigned tabs above

def _render_stock_comparison(tickers: list, market: str, key_prefix: str = ""):
    """并排对比2-4只股票的核心指标"""
    from data_fetcher import get_stock_data
    from indicator_utils import calculate_blue_signal_series, calculate_adx_series, calculate_heima_signal_series
    
    price_symbol = "¥" if market == "CN" else "$"
    n = len(tickers)
    cols = st.columns(n)
    
    for i, ticker in enumerate(tickers):
        with cols[i]:
            with st.spinner(f"加载 {ticker}..."):
                try:
                    hist = get_stock_data(ticker, market=market, days=365)
                    if hist is None or hist.empty:
                        st.error(f"❌ {ticker} 无数据")
                        continue
                    
                    # 计算指标
                    close = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else close
                    change_pct = (close / prev_close - 1) * 100
                    
                    # BLUE
                    try:
                        blue_vals = calculate_blue_signal_series(
                            hist['Open'].values, hist['High'].values,
                            hist['Low'].values, hist['Close'].values
                        )
                        blue = float(blue_vals[-1]) if len(blue_vals) > 0 else 0
                    except:
                        blue = 0
                    
                    # ADX
                    try:
                        adx_vals = calculate_adx_series(
                            hist['High'].values, hist['Low'].values, hist['Close'].values
                        )
                        adx = float(adx_vals[-1]) if len(adx_vals) > 0 else 0
                    except:
                        adx = 0
                    
                    # 黑马/掘地
                    try:
                        heima_result = calculate_heima_signal_series(
                            hist['Open'].values, hist['High'].values,
                            hist['Low'].values, hist['Close'].values, hist['Volume'].values
                        )
                        is_heima = bool(heima_result.get('is_heima', False))
                        is_juedi = bool(heima_result.get('is_juedi', False))
                    except:
                        is_heima, is_juedi = False, False
                    
                    # 近期表现
                    perf_5d = (close / float(hist['Close'].iloc[-6]) - 1) * 100 if len(hist) > 6 else 0
                    perf_20d = (close / float(hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 21 else 0
                    
                    # 波动率
                    if len(hist) > 20:
                        returns = hist['Close'].pct_change().dropna().tail(20)
                        volatility = float(returns.std() * (252 ** 0.5) * 100)
                    else:
                        volatility = 0
                    
                    # 成交量比
                    vol_avg = float(hist['Volume'].tail(20).mean()) if len(hist) > 20 else 0
                    vol_today = float(hist['Volume'].iloc[-1])
                    vol_ratio = vol_today / vol_avg if vol_avg > 0 else 0
                    
                    # 卡片展示
                    st.markdown(f"### {ticker}")
                    
                    # 价格 & 涨跌
                    delta_str = f"{change_pct:+.2f}%"
                    st.metric("价格", f"{price_symbol}{close:.2f}", delta_str)
                    
                    # 核心信号
                    blue_color = "🟢" if blue > 70 else "🟡" if blue > 50 else "🔴"
                    adx_color = "🟢" if adx > 25 else "🟡" if adx > 15 else "⚪"
                    st.markdown(f"{blue_color} **BLUE** {blue:.0f} &nbsp;&nbsp; {adx_color} **ADX** {adx:.0f}")
                    
                    signals = []
                    if is_heima: signals.append("🐴黑马")
                    if is_juedi: signals.append("⛏️掘地")
                    if signals:
                        st.markdown(" ".join(signals))
                    
                    # 表现对比
                    st.caption(f"5日: {perf_5d:+.1f}% | 20日: {perf_20d:+.1f}%")
                    st.caption(f"波动率: {volatility:.1f}% | 量比: {vol_ratio:.1f}")
                    
                    # 迷你K线 (最近30天收盘价走势)
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    recent = hist.tail(30)
                    fig.add_trace(go.Scatter(
                        x=recent.index, y=recent['Close'],
                        mode='lines', line=dict(width=2, color='#00d4aa'),
                        fill='tozeroy', fillcolor='rgba(0,212,170,0.1)'
                    ))
                    fig.update_layout(
                        height=120, margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"compare_chart_{key_prefix}_{ticker}")
                    
                    # 操作按钮
                    if st.button(f"🔍 详情", key=f"compare_detail_{key_prefix}_{ticker}", use_container_width=True):
                        st.session_state[f'compare_detail_{key_prefix}'] = ticker
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"{ticker}: {e}")
    
    # 对比总结
    st.divider()
    st.markdown("**📊 对比总结**: 选中多只股票可快速比较核心指标。点击「详情」深入分析单只股票。")
    
    # 如果点了详情按钮，展开该股票
    detail_key = f'compare_detail_{key_prefix}'
    if detail_key in st.session_state and st.session_state[detail_key]:
        detail_ticker = st.session_state[detail_key]
        st.divider()
        render_unified_stock_detail(
            symbol=detail_ticker,
            market=market,
            key_prefix=f"compare_{key_prefix}"
        )
        if st.button("← 返回对比", key=f"back_compare_{key_prefix}"):
            del st.session_state[detail_key]
            st.rerun()


def render_scan_page():
    st.header("🦅 每日机会扫描 (Opportunity Scanner)")
    
    # 侧边栏：数据源选择 (必须先执行，才能获得 market 值)
    with st.sidebar:
        st.divider()
        st.header("📂 数据源")
        
        # === 市场选择器 ===
        st.subheader("🌍 市场选择")
        market_options = {"🇺🇸 美股 (US)": "US", "🇨🇳 A股 (CN)": "CN"}
        selected_market_label = st.radio(
            "选择市场",
            options=list(market_options.keys()),
            horizontal=True,
            index=0,
            key="scan_market",
            help="切换美股/A股扫描结果"
        )
        selected_market = market_options[selected_market_label]
        _set_active_market(selected_market)
    
    # === Market Pulse Dashboard (顶部) - 传入选中的市场 ===
    render_market_pulse(market=selected_market)
    render_market_news_intel(market=selected_market, unique_key='scan')
    
    # 侧边栏：继续其他设置
    with st.sidebar:
        st.divider()
        
        # 检查数据库状态
        try:
            init_db()
            stats = _cached_db_stats()
            use_db = stats and stats['total_records'] > 0
        except:
            use_db = False
            stats = None
        
        if use_db:
            st.success("✅ 数据库模式")
            st.caption(f"📊 总记录: {stats['total_records']:,}")
            st.caption(f"📅 日期范围: {stats['min_date']} ~ {stats['max_date']}")
            
            # 日期选择器 - 按所选市场过滤
            available_dates = _cached_scanned_dates(market=selected_market)
            if available_dates:
                # 转换为 datetime 对象用于 selectbox
                date_options = available_dates[:30]  # 最近30天
                selected_date = st.selectbox(
                    "📅 选择日期",
                    options=date_options,
                    index=0,
                    help=f"选择要查看的 {selected_market} 扫描日期"
                )
                
                # 显示该日期的扫描状态
                try:
                    job = get_scan_job(selected_date)
                except Exception:
                    job = None
                if job:
                    st.caption(f"⏱️ 扫描于: {job.get('finished_at', 'N/A')}")
                    st.caption(f"📈 发现信号: {job.get('signals_found', 'N/A')} 只")
            else:
                selected_date = None
                st.warning(f"暂无 {selected_market} 扫描数据")
        else:
            st.info("📁 CSV 文件模式")
            selected_date = None
        
        if st.button("🔄 刷新数据"):
            st.rerun()
    
    # 加载数据 - 按所选市场过滤
    if use_db and selected_date:
        df, data_source = load_scan_results_from_db(selected_date, market=selected_market)
        if data_source:
            data_source = f"📅 {data_source} ({selected_market})"
    else:
        df, data_source = load_latest_scan_results()
        if data_source and not data_source.startswith("📅"):
            data_source = f"📁 {data_source}"
    
    # === 调试: 检查数据加载后的 Heima 列（仅在显式开启时输出） ===
    if (
        os.environ.get("APP_DEBUG_HEIMA", "false").lower() == "true"
        and df is not None
        and not df.empty
        and 'Heima_Daily' in df.columns
    ):
        heima_true_count = df['Heima_Daily'].sum()
        heima_sample = df['Heima_Daily'].head(5).tolist()
        heima_types = [type(v).__name__ for v in heima_sample]
        print(f"[DEBUG] 加载后 Heima_Daily: True={heima_true_count}/{len(df)}, 样本={heima_sample}, 类型={heima_types}")

    if df is None or df.empty:
        st.warning("⚠️ 未找到扫描结果。")
        
        # 调试：显示为什么没有数据
        with st.expander("🔍 调试信息", expanded=True):
            st.write(f"use_db: {use_db}, selected_date: {selected_date}, market: {selected_market}")
            if use_db and selected_date:
                raw_results = _cached_scan_results(scan_date=selected_date, market=selected_market)
                st.write(f"raw_results count: {len(raw_results) if raw_results else 0}")
                if raw_results:
                    st.write("第一条数据的 keys:", list(raw_results[0].keys())[:10])
                    st.write("第一条数据:", {k: v for k, v in list(raw_results[0].items())[:8]})
                # 也试试不带 market 参数
                raw_no_market = query_scan_results(scan_date=selected_date, limit=5)
                st.write(f"不带market查询: {len(raw_no_market) if raw_no_market else 0} 条")
                if raw_no_market:
                    st.write("market值:", [r.get('market') for r in raw_no_market[:5]])
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **方式一**: 运行每日扫描\n```bash\ncd versions/v2\npython scripts/run_daily_scan.py\n```")
        with col2:
            st.info("💡 **方式二**: 批量回填历史数据\n```bash\ncd versions/v2\npython scripts/backfill.py --start 2025-12-01 --end 2026-01-07\n```")
        return

    # === 价格口径统一：信号价(触发当日) + 现价(最新扫描) ===
    try:
        if 'Price' in df.columns and 'Ticker' in df.columns:
            df['信号价'] = pd.to_numeric(df['Price'], errors='coerce')
            latest_dates_for_market = _cached_scanned_dates(market=selected_market) or []
            latest_date_for_market = latest_dates_for_market[0] if latest_dates_for_market else selected_date
            latest_price_map = {}
            if latest_date_for_market:
                latest_rows = _cached_scan_results(
                    scan_date=latest_date_for_market,
                    market=selected_market,
                    limit=5000,
                ) or []
                for r in latest_rows:
                    sym = str(r.get('symbol') or '').upper().strip()
                    if not sym:
                        continue
                    px = pd.to_numeric(r.get('price'), errors='coerce')
                    if pd.notna(px):
                        latest_price_map[sym] = float(px)
            df['现价'] = df['Ticker'].map(lambda t: latest_price_map.get(str(t).upper().strip()))
            df['现价'] = pd.to_numeric(df['现价'], errors='coerce').fillna(df['信号价'])
            base_px = pd.to_numeric(df['信号价'], errors='coerce')
            curr_px = pd.to_numeric(df['现价'], errors='coerce')
            df['价格变化(%)'] = np.where(
                (base_px > 0) & np.isfinite(base_px) & np.isfinite(curr_px),
                (curr_px / base_px - 1.0) * 100.0,
                np.nan,
            )
    except Exception:
        # 价格增强仅影响展示，不阻断主流程
        pass
            
    # === Qlib 结果加载 (用于融合) ===
    def _load_qlib_latest_pack(market: str) -> dict:
        from pathlib import Path
        import json

        base = Path(current_dir) / "ml" / "saved_models" / f"qlib_{market.lower()}"
        out = {
            "available": False,
            "summary": {},
            "segment_df": pd.DataFrame(),
        }
        try:
            summary_path = base / "qlib_mining_summary_latest.json"
            seg_path = base / "segment_strategy_compare_latest.csv"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    out["summary"] = json.load(f)
            if seg_path.exists():
                out["segment_df"] = pd.read_csv(seg_path)
            out["available"] = bool(out["summary"] or not out["segment_df"].empty)
        except Exception:
            pass
        return out

    def _qlib_segment_to_caps(seg: str):
        seg = (seg or "").upper()
        if seg == "LARGE":
            return ["Mega-Cap (超大盘)", "Mega-Cap (巨头)", "Large-Cap (大盘)", "Large-Cap"]
        if seg == "MID":
            return ["Mid-Cap (中盘)", "Mid-Cap"]
        if seg == "SMALL":
            return ["Small-Cap (小盘)", "Small-Cap", "Micro-Cap (微盘)", "Micro-Cap"]
        return []

    qlib_pack = _load_qlib_latest_pack(selected_market)

    # === 🏆 智能排序 & Alpha Picks ===
    try:
        from ml.ranking_system import get_ranking_system
        ranker = get_ranking_system()
        df = ranker.calculate_integrated_score(df, scan_date=selected_date if 'selected_date' in dir() else None)
        
        if 'mmoe_dir_prob' in df.columns and df['mmoe_dir_prob'].notna().any():
            mmoe_valid = df['mmoe_dir_prob'].dropna()
            mmoe_count = len(mmoe_valid)
            avg_dir = mmoe_valid.mean()
            bullish_pct = (mmoe_valid > 0.4).mean() * 100
            bearish_pct = (mmoe_valid < 0.15).mean() * 100
            
            # 市场温度计
            if avg_dir > 0.55:
                temp_emoji, temp_label, temp_color = "🟢", "多头市场", "green"
            elif avg_dir > 0.45:
                temp_emoji, temp_label, temp_color = "🟡", "震荡市", "orange"
            elif avg_dir > 0.35:
                temp_emoji, temp_label, temp_color = "🟠", "偏空", "orange"
            else:
                temp_emoji, temp_label, temp_color = "🔴", "空头市场", "red"
            
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric(f"{temp_emoji} 市场温度", temp_label, f"均值 {avg_dir:.1%}")
            tc2.metric("📊 AI 覆盖", f"{mmoe_count}/{len(df)}", f"{mmoe_count/len(df)*100:.0f}%")
            tc3.metric("🟢 偏多信号", f"{bullish_pct:.0f}%", f"dir>0.4")
            tc4.metric("🔴 偏空信号", f"{bearish_pct:.0f}%", f"dir<0.15")
            
            # 添加信号方向列
            def _signal_dir(p):
                if pd.isna(p): return "⚪观望"
                if p >= 0.4: return "🟢做多"
                if p >= 0.25: return "⚪观望"
                if p >= 0.15: return "🟠谨慎"
                return "🔴做空"
            df['信号方向'] = df['mmoe_dir_prob'].apply(_signal_dir)
    except Exception:
        pass

    # 侧边栏：继续筛选器
    with st.sidebar:
        st.divider()
        st.header("🎛️ 多维筛选")
        st.caption("根据您的偏好自由组合过滤条件")

        # === Qlib 融合层 ===
        qlib_blend_enabled = False
        qlib_focus_segment = None
        qlib_best_topk = None
        if qlib_pack["available"]:
            with st.expander("🧠 Qlib 融合", expanded=False):
                seg_df = qlib_pack.get("segment_df", pd.DataFrame())
                summary = qlib_pack.get("summary", {})

                if not seg_df.empty and "best_sharpe" in seg_df.columns:
                    tmp = seg_df.sort_values("best_sharpe", ascending=False).iloc[0]
                    qlib_focus_segment = str(tmp.get("segment", ""))
                    qlib_best_topk = int(tmp.get("best_topk", 8) or 8)
                    st.caption(
                        f"建议分层: **{qlib_focus_segment}** | "
                        f"建议持仓: **Top {qlib_best_topk}** | "
                        f"Sharpe: **{float(tmp.get('best_sharpe', 0)):.2f}**"
                    )
                else:
                    top = (summary.get("top_strategies") or [{}])[0]
                    qlib_best_topk = int(top.get("topk", 8) or 8)
                    qlib_focus_segment = summary.get("segment", "")
                    st.caption(
                        f"建议分层: **{qlib_focus_segment or 'N/A'}** | "
                        f"建议持仓: **Top {qlib_best_topk}**"
                    )

                qlib_blend_enabled = st.checkbox(
                    "应用 Qlib 融合过滤与打分",
                    value=False,
                    help="将 Qlib 挖掘结果与当前扫描特征融合：按分层过滤 + 融合分排序。",
                )
        else:
            st.caption("🧠 Qlib 融合：暂无可用结果文件")
        
        # === 1. 流动性筛选 (最重要!) ===
        st.subheader("💧 流动性")
        _filter_steps = [(f"加载后", len(df))]
        
        # 日均成交额 (Turnover) - 使用 Turnover_M 列 (百万美元)
        if 'Turnover' in df.columns:
            turnover_col = 'Turnover'
        elif 'Turnover_M' in df.columns:
            df['Turnover'] = df['Turnover_M']  # 统一列名
            turnover_col = 'Turnover'
        else:
            turnover_col = None
            
        if turnover_col and turnover_col in df.columns:
            max_turnover = float(df[turnover_col].max()) if df[turnover_col].max() > 0 else 1000
            
            # 快捷按钮
            st.caption("快捷筛选:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("≥1M", key="t1m", help="成交额≥100万"):
                    st.session_state['turnover_filter'] = 1.0
            with col2:
                if st.button("≥5M", key="t5m", help="成交额≥500万"):
                    st.session_state['turnover_filter'] = 5.0
            with col3:
                if st.button("≥10M", key="t10m", help="成交额≥1000万"):
                    st.session_state['turnover_filter'] = 10.0
            
            # 获取筛选值
            default_val = st.session_state.get('turnover_filter', 0.5)
            
            min_turnover_val = st.slider(
                "最低日成交额 (百万)", 
                min_value=0.0, 
                max_value=min(max_turnover, 100.0),
                value=min(default_val, max_turnover),
                step=0.5,
                help="过滤成交额过低的股票，避免流动性风险。1M=100万"
            )
            st.session_state['turnover_filter'] = min_turnover_val
            _tv = pd.to_numeric(df[turnover_col], errors='coerce')
            df = df[(_tv >= min_turnover_val) | _tv.isna()]
            _filter_steps.append((f"流动性≥{min_turnover_val}M", len(df)))
        
        # === 2. 信号强度筛选 ===
        st.subheader("📊 信号强度")
        
        # BLUE 信号
        if 'Day BLUE' in df.columns:
            blue_range = st.slider(
                "Day BLUE 范围",
                min_value=0.0,
                max_value=200.0,
                value=(0.0, 200.0),  # 默认 0-200 (显示所有)
                step=10.0,
                help="BLUE 越高代表抄底信号越强"
            )
            _blue = pd.to_numeric(df['Day BLUE'], errors='coerce')
            df = df[_blue.between(blue_range[0], blue_range[1]) | _blue.isna()]
            _filter_steps.append((f"BLUE {blue_range[0]}-{blue_range[1]}", len(df)))
        
        # ADX 趋势强度
        if 'ADX' in df.columns:
            adx_min = st.slider(
                "最低 ADX (趋势强度)",
                min_value=0.0,
                max_value=80.0,
                value=0.0,  # 默认 0 (显示所有)
                step=5.0,
                help="ADX > 25 表示趋势明确，ADX > 40 表示强趋势"
            )
            _adx = pd.to_numeric(df['ADX'], errors='coerce')
            df = df[(_adx >= adx_min) | _adx.isna()]
            _filter_steps.append((f"ADX≥{adx_min}", len(df)))
        
        # === 3. 市值与价格筛选 ===
        st.subheader("💰 市值 & 价格")
        
        # 市值规模 (Multi-Select)
        if 'Cap_Category' in df.columns:
            all_caps = df['Cap_Category'].unique().tolist()
            # 排序：按市值从大到小
            cap_order = ['Mega-Cap (巨头)', 'Large-Cap', 'Mid-Cap', 'Small-Cap', 'Micro-Cap', 'Unknown']
            sorted_caps = [c for c in cap_order if c in all_caps] + [c for c in all_caps if c not in cap_order]
            selected_caps = st.multiselect(
                "市值规模", 
                sorted_caps, 
                default=sorted_caps,
                help="Mega > $200B, Large > $10B, Mid > $2B, Small > $300M, Micro < $300M"
            )
            if selected_caps:
                df = df[df['Cap_Category'].isin(selected_caps)]
                _filter_steps.append(("市值筛选", len(df)))
        
        # 价格区间
        if 'Price' in df.columns:
            price_range = st.slider(
                "价格区间 ($)",
                min_value=0.0,
                max_value=min(float(df['Price'].max()), 5000.0),
                value=(1.0, 1000.0),  # 默认 $1-$1000
                step=1.0,
                help="过滤仙股 (<$1) 和超高价股"
            )
            _px = pd.to_numeric(df['Price'], errors='coerce')
            df = df[_px.between(price_range[0], price_range[1]) | _px.isna()]
            _filter_steps.append((f"价格${price_range[0]}-${price_range[1]}", len(df)))
        
        # === 4. 策略类型筛选 ===
        st.subheader("🎯 策略类型")
        
        if 'Strategy' in df.columns:
            all_strategies = df['Strategy'].unique().tolist()
            selected_strategies = st.multiselect(
                "策略标签", 
                all_strategies, 
                default=all_strategies,
                help="Trend-D: 趋势跟随, Resonance-C: 多周期共振"
            )
            if selected_strategies:
                df = df[df['Strategy'].isin(selected_strategies)]
        
        # === 5. 黑马/掘地信号筛选 ===
        st.subheader("🐴⛏️ 信号筛选")
        
        # 初始化 session_state
        if 'heima_filter' not in st.session_state:
            st.session_state['heima_filter'] = '全部'
        
        signal_options = ["全部", "有日黑马", "有周黑马", "有月黑马", "有任意黑马",
                          "有日掘地", "有周掘地", "有月掘地", "有任意掘地"]
        
        heima_filter = st.radio(
            "信号筛选",
            options=signal_options,
            horizontal=True,
            help="筛选出有黑马🐴或掘地⛏️信号的股票",
            key="heima_filter"
        )
        
        # === 6. 高级筛选 (折叠) ===
        with st.expander("🔬 高级筛选", expanded=False):
            # 获利盘比例
            if 'Profit_Ratio' in df.columns:
                pr_range = st.slider(
                    "获利盘比例 (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),  # 默认不限制
                    step=5,
                    help="获利盘高 = 筹码结构好，但可能已经涨过；获利盘低 = 套牢盘多，反弹空间大但风险也大"
                )
                pr_vals = pd.to_numeric(df['Profit_Ratio'], errors='coerce') * 100
                # NaN 行保留（数据缺失不应被过滤掉）
                df = df[pr_vals.between(pr_range[0], pr_range[1]) | pr_vals.isna()]
            
            # 波浪形态筛选
            if 'Wave_Phase' in df.columns:
                # NaN 填充为 "未知"，让用户可以看到并选择
                df['Wave_Phase'] = df['Wave_Phase'].fillna('未知')
                all_waves = df['Wave_Phase'].unique().tolist()
                selected_waves = st.multiselect("波浪形态", all_waves, default=all_waves)
                if selected_waves:
                    df = df[df['Wave_Phase'].isin(selected_waves)]
            
            # 缠论信号筛选
            if 'Chan_Signal' in df.columns:
                df['Chan_Signal'] = df['Chan_Signal'].fillna('未知')
                all_chans = df['Chan_Signal'].unique().tolist()
                selected_chans = st.multiselect("缠论信号", all_chans, default=all_chans)
                if selected_chans:
                    df = df[df['Chan_Signal'].isin(selected_chans)]
            
            # 筹码形态筛选 (需要先计算筹码)
            st.caption("💡 勾选下方「计算筹码形态」后可使用筹码筛选")
            chip_filter = st.selectbox(
                "🔥 筹码形态筛选",
                options=["全部", "仅强势顶格峰 🔥", "仅底部密集 📍", "有底部信号 (🔥+📍)"],
                index=0,
                help="需要先启用筹码形态计算"
            )
            # 存储到 session_state 供后续使用
            st.session_state['chip_filter'] = chip_filter
        
        _filter_steps.append(("筛选完成", len(df)))
        
        # 显示筛选结果统计
        st.divider()
        st.metric("筛选后结果", f"{len(df)} 只", help="符合所有筛选条件的股票数量")

    # === 应用 Qlib 融合 ===
    qlib_impact = ""
    if qlib_pack["available"] and qlib_blend_enabled:
        before_n = len(df)

        # 1) 按分层优先过滤
        target_caps = _qlib_segment_to_caps(qlib_focus_segment)
        if target_caps and "Cap_Category" in df.columns:
            filtered = df[df["Cap_Category"].isin(target_caps)]
            if len(filtered) >= 3:
                df = filtered

        # 2) 构建融合分（现有综合分 + 信号强度 + 趋势强度）
        # 注意：Qlib 当前输出是组合层结论，这里做的是交易执行层的可落地融合，而非逐票 qlib 预测。
        if "Integrated_Score" in df.columns:
            base = pd.to_numeric(df["Integrated_Score"], errors="coerce").fillna(0)
        elif "Score" in df.columns:
            base = pd.to_numeric(df["Score"], errors="coerce").fillna(0)
        else:
            base = pd.Series(0, index=df.index, dtype=float)

        blue = pd.to_numeric(df.get("Day BLUE", 0), errors="coerce").fillna(0)
        adx = pd.to_numeric(df.get("ADX", 0), errors="coerce").fillna(0)
        qlib_bonus = 10 if qlib_focus_segment else 0
        df["Qlib_Fusion_Score"] = base * 0.65 + blue * 0.20 + adx * 0.15 + qlib_bonus
        df = df.sort_values("Qlib_Fusion_Score", ascending=False)

        # 3) 按建议持仓数给出候选池（TopK * 3）
        if qlib_best_topk and qlib_best_topk > 0 and len(df) > qlib_best_topk * 3:
            df = df.head(qlib_best_topk * 3)

        qlib_impact = f"Qlib融合已生效: {before_n} → {len(df)} 只"

    # [调试] 显示数据加载和筛选情况
    with st.expander("🔍 调试: 数据加载状态", expanded=(df is None or (hasattr(df, 'empty') and df.empty))):
        try:
            raw = _cached_scan_results(scan_date=selected_date, market=selected_market) if selected_date else None
            raw_count = len(raw) if raw else 0
        except Exception:
            raw_count = -1
        filtered_count = len(df) if df is not None and hasattr(df, '__len__') else 0
        st.write(f"📊 原始查询: {raw_count} 条 → 筛选后: {filtered_count} 条")
        st.write(f"📅 日期: {selected_date} | 市场: {selected_market}")
        if df is not None and not df.empty:
            st.write(f"列名: {list(df.columns[:15])}")
        elif raw_count > 0:
            st.warning(f"⚠️ 原始有 {raw_count} 条但 df 为空！")
        # 显示筛选步骤跟踪
        if '_filter_steps' in dir():
            pass
        try:
            st.write("📉 筛选步骤:")
            for step_name, step_count in _filter_steps:
                st.write(f"  {step_name}: {step_count} 条")
        except Exception:
            pass

    # 2. 顶部仪表盘
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        label = "筛选后机会"
        if qlib_impact:
            label = "筛选后机会 (Qlib融合)"
        st.metric(label, f"{len(df)} 只", help="符合当前筛选条件的股票数量")

    with col2:
        # 强信号：BLUE > 150
        strong_signals = len(df[df['Day BLUE'] > 150]) if 'Day BLUE' in df.columns else 0
        st.metric("🔥 强信号 (BLUE>150)", f"{strong_signals} 只", help="BLUE > 150 的强势抄底信号")

    with col3:
        trend_opps = len(df[df['Strategy'].str.contains('Trend', na=False)]) if 'Strategy' in df.columns else 0
        st.metric("🚀 趋势突破", f"{trend_opps} 只", help="Strategy D: 趋势跟随")

    with col4:
        # 高流动性：成交额 > 10M
        if 'Turnover' in df.columns:
            high_liquidity = len(df[df['Turnover'] > 10])
            st.metric("💧 高流动性 (>$10M)", f"{high_liquidity} 只", help="日成交额 > 1000万美元")
        else:
            mood, color = get_market_mood(df)
            st.markdown(f"**市场情绪**")
            st.markdown(f"<h3 style='color: {color}; margin-top: -10px;'>{mood}</h3>", unsafe_allow_html=True)

    st.divider()
    if qlib_impact:
        st.info(f"🧠 {qlib_impact}")

    # 3. 机会清单
    st.subheader("📋 机会清单 (Opportunity Matrix)")
    
    # 底部顶格峰计算选项
    col_opt1, col_opt2 = st.columns([1, 3])
    with col_opt1:
        calc_chip = st.checkbox("🔥 计算筹码形态", value=False, help="计算底部顶格峰 (首次约 30-60 秒，后续使用缓存)")
    
    # 使用 session_state 缓存结果
    cache_key = f"chip_cache_{selected_date}_{selected_market}"
    
    if calc_chip:
        # 检查缓存
        if cache_key in st.session_state:
            cached_data = st.session_state[cache_key]
            # 验证缓存是否包含当前所有股票
            cached_tickers = set(cached_data.keys())
            current_tickers = set(df['Ticker'].tolist())
            if current_tickers.issubset(cached_tickers):
                # 使用缓存
                chip_labels = [cached_data.get(t, '') for t in df['Ticker'].tolist()]
                df['筹码形态'] = chip_labels
                strong_peaks = chip_labels.count('🔥')
                normal_peaks = chip_labels.count('📍')
                st.caption(f"⚡ 使用缓存 | 🔥 强势: {strong_peaks} | 📍 底部密集: {normal_peaks}")
            else:
                # 缓存不完整，需要重新计算
                st.session_state.pop(cache_key, None)
                st.rerun()
        else:
            # 并行计算
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            tickers = df['Ticker'].tolist()
            results = {}
            
            def calc_single(ticker):
                try:
                    stock_df = fetch_data_from_polygon(ticker, days=100)
                    if stock_df is not None and len(stock_df) >= 30:
                        result = quick_chip_analysis(stock_df)
                        return ticker, result.get('label', '') if result else ''
                    return ticker, ''
                except:
                    return ticker, ''
            
            progress_bar = st.progress(0, text="正在分析筹码分布...")
            
            # 使用线程池并行计算 (最多 10 个并发)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(calc_single, t): t for t in tickers}
                completed = 0
                for future in as_completed(futures):
                    ticker, label = future.result()
                    results[ticker] = label
                    completed += 1
                    progress_bar.progress(completed / len(tickers), 
                                          text=f"分析中 {completed}/{len(tickers)} ({ticker})")
            
            progress_bar.empty()
            
            # 保存到缓存
            st.session_state[cache_key] = results
            
            chip_labels = [results.get(t, '') for t in tickers]
            df['筹码形态'] = chip_labels
            
            strong_peaks = chip_labels.count('🔥')
            normal_peaks = chip_labels.count('📍')
            if strong_peaks > 0 or normal_peaks > 0:
                st.success(f"✅ 分析完成！🔥 强势顶格峰: {strong_peaks} 只 | 📍 底部密集: {normal_peaks} 只")
        
        # 应用筹码筛选器
        chip_filter = st.session_state.get('chip_filter', '全部')
        if '筹码形态' in df.columns and chip_filter != '全部':
            before_count = len(df)
            if chip_filter == "仅强势顶格峰 🔥":
                df = df[df['筹码形态'] == '🔥']
            elif chip_filter == "仅底部密集 📍":
                df = df[df['筹码形态'] == '📍']
            elif chip_filter == "有底部信号 (🔥+📍)":
                df = df[df['筹码形态'].isin(['🔥', '📍'])]
            st.info(f"📊 筹码筛选: {before_count} → {len(df)} 只")

    column_config = {
        "Ticker": st.column_config.TextColumn("代码", help="股票代码", width="small"),
        "Name": st.column_config.TextColumn("名称", width="medium"),
        "Mkt Cap": st.column_config.NumberColumn("市值 ($B)", format="%.2f", help="市值 (十亿美元)"),
        "Price": st.column_config.NumberColumn("现价", format="$%.2f"),
        "信号价": st.column_config.NumberColumn("信号价", format="$%.2f", help="该信号触发当日价格"),
        "现价": st.column_config.NumberColumn("现价", format="$%.2f", help="最新扫描日价格"),
        "价格变化(%)": st.column_config.NumberColumn("价格变化", format="%.2f%%", help="现价相对信号价变化"),
        "Turnover": st.column_config.NumberColumn("成交额 ($M)", format="%.1f", help="日成交额 (百万美元)"),
        "Day BLUE": st.column_config.ProgressColumn(
            "日 BLUE", format="%.0f", min_value=0, max_value=200,
            help="日线抄底信号强度 (0-200)"
        ),
        "Week BLUE": st.column_config.ProgressColumn(
            "周 BLUE", format="%.0f", min_value=0, max_value=200,
            help="周线抄底信号强度 (0-200)"
        ),
        "Month BLUE": st.column_config.ProgressColumn(
            "月 BLUE", format="%.0f", min_value=0, max_value=200,
            help="月线抄底信号强度 (0-200)"
        ),
        "ADX": st.column_config.NumberColumn("ADX", format="%.1f", help="趋势强度 (>25 趋势明确, >40 强趋势)"),
        "Strategy": st.column_config.TextColumn("策略标签", width="medium"),
        "Regime": st.column_config.TextColumn("波动属性", width="small"),
        "Cap_Category": st.column_config.TextColumn("市值规模", width="small"),
        "Stop Loss": st.column_config.NumberColumn("止损价", format="$%.2f", help="建议止损位"),
        "Shares Rec": st.column_config.NumberColumn("建议仓位", format="%d 股", help="基于$1000风险敞口的建议股数"),
        "Wave_Desc": st.column_config.TextColumn("波浪形态", width="medium", help="Elliott Wave"),
        "Chan_Desc": st.column_config.TextColumn("缠论形态", width="medium", help="Chan Theory"),
        "Profit_Ratio": st.column_config.NumberColumn("获利盘", format="%.0f%%", help="获利盘比例"),
        "筹码形态": st.column_config.TextColumn("筹码", width="small", help="🔥=强势顶格峰 📍=底部密集"),
        "新发现": st.column_config.TextColumn("状态", width="small", help="🆕=今日新发现, 📅=之前出现过"),
        "历史信号": st.column_config.TextColumn("历史信号", width="medium", help="之前出过信号的日期"),
        "新闻": st.column_config.TextColumn("新闻", width="small", help="🟢利好/🔴利空 (利好数/利空数)")
    }
    if "Qlib_Fusion_Score" in df.columns:
        column_config["Qlib_Fusion_Score"] = st.column_config.NumberColumn(
            "Qlib融合分",
            format="%.1f",
            help="Qlib 组合结论与现有技术特征融合后的执行分",
        )

    # === 新发现标记 + 历史信号 ===
    if 'Ticker' in df.columns and len(df) > 0:
        tickers = df['Ticker'].tolist()
        
        # 先获取历史信号日期（一次查完，两列共用）
        try:
            from db.supabase_db import get_signal_history_dates_supabase
            history_dates = get_signal_history_dates_supabase(tickers, market=selected_market, limit_per_stock=30)
        except:
            history_dates = {}
        
        # 状态列：基于上一次信号日期（不是首次）
        def get_newness_label(ticker):
            dates = history_dates.get(ticker, [])
            prev_dates = [d for d in dates if d != selected_date]
            if not prev_dates:
                return "🆕新发现"
            
            last_date = prev_dates[0]  # 降序，第一个就是上一次
            try:
                last_dt = datetime.strptime(last_date, '%Y-%m-%d')
                selected_dt = datetime.strptime(selected_date, '%Y-%m-%d')
                days_diff = (selected_dt - last_dt).days
                if days_diff <= 0:
                    return "🆕新发现"
                return f"📅{days_diff}天前"
            except:
                return "📅"
        
        df['新发现'] = df['Ticker'].apply(get_newness_label)

        # 历史信号列：显示之前出过信号的日期
        def format_history(ticker):
            dates = history_dates.get(ticker, [])
            dates = [d for d in dates if d != selected_date]
            if not dates:
                return ""
            short = []
            for d in dates[:5]:
                try:
                    dt = datetime.strptime(d, '%Y-%m-%d')
                    short.append(f"{dt.month}/{dt.day}")
                except:
                    short.append(d[-5:])
            suffix = f" +{len(dates)-5}" if len(dates) > 5 else ""
            return ", ".join(short) + suffix
        
        df['历史信号'] = df['Ticker'].apply(format_history)

        # 保留 first_dates 用于信号价
        first_dates = get_first_scan_dates(tickers, market=selected_market)
        df['信号日期'] = df['Ticker'].map(first_dates)

        # 用“首次信号日价格”覆盖信号价，避免与现价同值导致无信息量
        try:
            from db.database import get_connection

            if tickers:
                conn = get_connection()
                cursor = conn.cursor()
                placeholders = ",".join(["?"] * len(tickers))
                cursor.execute(
                    f"""
                    SELECT symbol, scan_date, price
                    FROM scan_results
                    WHERE market = ?
                      AND symbol IN ({placeholders})
                      AND price IS NOT NULL
                    ORDER BY symbol ASC, scan_date ASC
                    """,
                    [selected_market] + tickers,
                )
                rows_px = cursor.fetchall()
                conn.close()

                first_row_px = {}
                first_signal_px = {}
                for rr in rows_px:
                    sym = str(rr["symbol"])
                    sdt = str(rr["scan_date"])
                    px = pd.to_numeric(rr["price"], errors="coerce")
                    if pd.isna(px):
                        continue
                    if sym not in first_row_px:
                        first_row_px[sym] = float(px)
                    if sym in first_dates and sdt == str(first_dates.get(sym)) and sym not in first_signal_px:
                        first_signal_px[sym] = float(px)

                signal_px_map = {
                    sym: first_signal_px.get(sym, first_row_px.get(sym))
                    for sym in tickers
                }
                df['信号价'] = df['Ticker'].map(signal_px_map).fillna(df.get('信号价', df.get('Price')))

                # 重新计算价格变化(%)，口径=现价相对首次信号价
                if '现价' in df.columns:
                    sig_px = pd.to_numeric(df['信号价'], errors='coerce')
                    cur_px = pd.to_numeric(df['现价'], errors='coerce')
                    df['价格变化(%)'] = np.where(
                        (sig_px > 0) & np.isfinite(sig_px) & np.isfinite(cur_px),
                        (cur_px / sig_px - 1.0) * 100.0,
                        np.nan,
                    )
        except Exception:
            pass




    # === 新闻情绪分析 ===
    # 添加新闻情绪列 (按需加载)
    news_cache_key = f"news_sentiment_{selected_date}_{selected_market}"
    
    col_news1, col_news2 = st.columns([1, 4])
    with col_news1:
        analyze_news = st.button("📰 获取新闻情绪", help="分析前10只股票的新闻情绪")
    with col_news2:
        if news_cache_key in st.session_state:
            cached_count = len([v for v in st.session_state[news_cache_key].values() if v])
            st.caption(f"✅ 已缓存 {cached_count} 只股票的新闻情绪")
    
    if analyze_news and 'Ticker' in df.columns and len(df) > 0:
        try:
            from news import get_news_intelligence
            intel = get_news_intelligence(use_llm=False)
            
            # 只分析前10只 (避免太慢)
            tickers_to_analyze = df['Ticker'].tolist()[:10]
            news_results = {}
            
            progress = st.progress(0, text="正在分析新闻...")
            for i, ticker in enumerate(tickers_to_analyze):
                try:
                    events, impacts, digest = intel.analyze_symbol(ticker, market=selected_market)
                    
                    if digest.total_news_count > 0:
                        ratio = digest.sentiment_ratio()
                        if ratio > 0.3:
                            emoji = "🟢"
                        elif ratio < -0.3:
                            emoji = "🔴"
                        else:
                            emoji = "⚪"
                        
                        news_results[ticker] = f"{emoji}{digest.bullish_count}/{digest.bearish_count}"
                    else:
                        news_results[ticker] = "➖"
                except:
                    news_results[ticker] = "❓"
                
                progress.progress((i + 1) / len(tickers_to_analyze), 
                                 text=f"分析 {ticker} ({i+1}/{len(tickers_to_analyze)})")
            
            progress.empty()
            
            # 缓存结果
            st.session_state[news_cache_key] = news_results
            st.success(f"✅ 新闻分析完成！{len(news_results)} 只股票")
            st.rerun()
            
        except Exception as e:
            st.error(f"新闻分析失败: {e}")
    
    # 显示列顺序：核心指标在前，新发现标记靠前，新闻情绪列
    if news_cache_key in st.session_state and 'Ticker' in df.columns:
        news_data = st.session_state[news_cache_key]
        df['新闻'] = df['Ticker'].map(lambda t: news_data.get(t, '➖'))

    # === 大师策略深度分析 ===
    master_cache_key = f"master_analysis_{selected_date}_{selected_market}"
    master_details_key = f"{master_cache_key}_details"
    
    col_master1, col_master2, col_master3 = st.columns([1, 3, 2])
    with col_master1:
        analyze_master = st.button("🤖 大师深度分析", help="基于5位大师策略分析前20只股票 (需获取历史数据，较慢)")
    with col_master2:
        if master_cache_key in st.session_state:
            cached_master = len([v for v in st.session_state[master_cache_key].values() if v])
            st.caption(f"✅ 已生成 {cached_master} 份大师报告")
    with col_master3:
        master_profile = st.selectbox(
            "策略偏好",
            options=["short", "medium", "long"],
            index=["short", "medium", "long"].index(st.session_state.get("master_profile", "medium")),
            key="master_profile_selector_scan",
            help="short=偏交易, medium=平衡, long=偏中长线稳健"
        )
        st.session_state["master_profile"] = master_profile

    if analyze_master and 'Ticker' in df.columns and len(df) > 0:
        try:
            from strategies.master_strategies import analyze_stock_for_master, get_master_summary_for_stock
            if selected_market == 'US':
                from data_fetcher import get_us_stock_data as get_data
            else:
                from data_fetcher import get_cn_stock_data as get_data
            
            # 先去重
            all_tickers = df['Ticker'].unique().tolist()
            # 分析前20只 (避免超时)
            tickers_to_analyze = all_tickers[:20]
            master_results = {}
            master_details = {} # 存储详细报告用于展示
            
            progress = st.progress(0, text="正在进行大师级推演...")
            
            for i, ticker in enumerate(tickers_to_analyze):
                try:
                    # 1. 获取近期历史数据 (用于计算均线、量比、九转)
                    hist_df = get_data(ticker, days=40)
                    
                    if hist_df is not None and not hist_df.empty:
                        # 准备参数
                        current_row = df[df['Ticker'] == ticker].iloc[0]
                        price = float(current_row.get('Price', 0))
                        
                        # 计算技术指标
                        sma5 = hist_df['Close'].rolling(5).mean().iloc[-1]
                        sma20 = hist_df['Close'].rolling(20).mean().iloc[-1]
                        
                        # 量比
                        vol = hist_df['Volume'].iloc[-1]
                        vol_ma5 = hist_df['Volume'].rolling(5).mean().iloc[-1]
                        vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0
                        
                        # 九转计数 (简单计算)
                        close_prices = hist_df['Close'].values
                        td_count = 0
                        if len(close_prices) > 13:
                            # 简化的TD检测，实际应使用 SignalDetector
                            c = close_prices
                            if c[-1] < c[-5]: # 下跌
                                count = 0
                                for k in range(1, 10):
                                    if c[-k] < c[-k-4]: count -= 1
                                    else: break
                                td_count = count
                            elif c[-1] > c[-5]: # 上涨
                                count = 0
                                for k in range(1, 10):
                                    if c[-k] > c[-k-4]: count += 1
                                    else: break
                                td_count = count
                        
                        # 2. 调用大师分析
                        analyses = analyze_stock_for_master(
                            symbol=ticker,
                            blue_daily=float(current_row.get('Day BLUE', 0)),
                            blue_weekly=float(current_row.get('Week BLUE', 0)),
                            blue_monthly=float(current_row.get('Month BLUE', 0)),
                            adx=float(current_row.get('ADX', 0)),
                            vol_ratio=vol_ratio,
                            change_pct=float(hist_df['Close'].pct_change().iloc[-1] * 100),
                            price=price,
                            sma5=sma5,
                            sma20=sma20,
                            td_count=td_count,
                            is_heima=True if '黑马' in str(current_row.get('Strategy', '')) else False
                        )
                        
                        # 3. 汇总结果
                        summary = get_master_summary_for_stock(
                            analyses,
                            profile=st.session_state.get("master_profile", "medium")
                        )
                        
                        # 存入结果
                        master_results[ticker] = summary['overall_action']
                        master_details[ticker] = analyses
                        
                    else:
                        master_results[ticker] = "数据不足"
                        
                except Exception as e:
                    master_results[ticker] = "分析失败"
                    print(f"Error analyzing {ticker}: {e}")
                
                progress.progress((i + 1) / len(tickers_to_analyze), 
                                 text=f"大师正在分析 {ticker} ({i+1}/{len(tickers_to_analyze)})")
            
            progress.empty()
            
            # 缓存结果
            st.session_state[master_cache_key] = master_results
            st.session_state[master_details_key] = master_details
            st.success(f"✅ 大师分析完成！已生成 {len(master_results)} 份策略报告")
            st.rerun()
            
        except Exception as e:
            st.error(f"大师分析服务异常: {e}")
            import traceback
            st.code(traceback.format_exc())

    # 将大师建议合并到 DataFrame
    if master_cache_key in st.session_state and 'Ticker' in df.columns:
        master_data = st.session_state[master_cache_key]
        df['大师建议'] = df['Ticker'].map(lambda t: master_data.get(t, '➖'))

    # 更新列配置
    column_config.update({
        "大师建议": st.column_config.TextColumn("大师建议", width="medium", help="5位大师综合评级")
    })

    # === 添加黑马列 (修复版) ===
    # 检测黑马字段
    def get_col(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None
    
    def safe_bool_convert(series):
        """
        安全地将列转换为布尔值
        处理: 0/1, True/False, None, bytes (b'\x01'), strings ('True'/'False')
        """
        import numpy as np
        
        def to_bool(val):
            # 1. 处理 None 和 NaN
            if val is None:
                return False
            try:
                if pd.isna(val):
                    return False
            except (TypeError, ValueError):
                pass  # 某些类型不支持 pd.isna
            
            # 2. 处理布尔值 (包括 numpy bool)
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            
            # 3. 处理整数/浮点数
            if isinstance(val, (int, float, np.integer, np.floating)):
                return val == 1  # 只有 1 才是 True
            
            # 4. 处理字节 (SQLite BLOB)
            if isinstance(val, bytes):
                return val == b'\x01'
            
            # 5. 处理字符串 (Supabase JSON 可能返回字符串)
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 't')
            
            # 6. 未知类型，默认 False
            print(f"[DEBUG] safe_bool_convert: 未知类型 {type(val).__name__}: {val!r}")
            return False
        
        return series.apply(to_bool)
    
    heima_daily_col = get_col(df, ['Heima_Daily', 'heima_daily'])
    heima_weekly_col = get_col(df, ['Heima_Weekly', 'heima_weekly'])
    heima_monthly_col = get_col(df, ['Heima_Monthly', 'heima_monthly'])
    heima_any_col = get_col(df, ['Is_Heima', 'is_heima'])  # 兼容旧数据
    juedi_daily_col = get_col(df, ['Juedi_Daily', 'juedi_daily'])
    juedi_weekly_col = get_col(df, ['Juedi_Weekly', 'juedi_weekly'])
    juedi_monthly_col = get_col(df, ['Juedi_Monthly', 'juedi_monthly'])
    juedi_any_col = get_col(df, ['Is_Juedi', 'is_juedi'])
    
    # === 黑马布尔列 ===
    if heima_daily_col:
        df['日黑马'] = safe_bool_convert(df[heima_daily_col])
    elif heima_any_col:
        df['日黑马'] = safe_bool_convert(df[heima_any_col])
    else:
        df['日黑马'] = False
    
    if heima_weekly_col:
        df['周黑马'] = safe_bool_convert(df[heima_weekly_col])
    else:
        df['周黑马'] = False
    
    if heima_monthly_col:
        df['月黑马'] = safe_bool_convert(df[heima_monthly_col])
    else:
        df['月黑马'] = False
    
    # === 掘地布尔列 ===
    if juedi_daily_col:
        df['日掘地'] = safe_bool_convert(df[juedi_daily_col])
    elif juedi_any_col:
        df['日掘地'] = safe_bool_convert(df[juedi_any_col])
    else:
        df['日掘地'] = False
    
    if juedi_weekly_col:
        df['周掘地'] = safe_bool_convert(df[juedi_weekly_col])
    else:
        df['周掘地'] = False
    
    if juedi_monthly_col:
        df['月掘地'] = safe_bool_convert(df[juedi_monthly_col])
    else:
        df['月掘地'] = False
    
    # 显示列 (图标)
    df['日🐴'] = df['日黑马'].apply(lambda x: '🐴' if x else '')
    df['周🐴'] = df['周黑马'].apply(lambda x: '🐴' if x else '')
    df['月🐴'] = df['月黑马'].apply(lambda x: '🐴' if x else '')
    df['日⛏️'] = df['日掘地'].apply(lambda x: '⛏️' if x else '')
    df['周⛏️'] = df['周掘地'].apply(lambda x: '⛏️' if x else '')
    df['月⛏️'] = df['月掘地'].apply(lambda x: '⛏️' if x else '')

    # 文本信号标签（清晰标注触发类型）
    def _build_signal_text(row):
        labels = []
        if bool(row.get('日黑马')):
            labels.append('日黑马')
        if bool(row.get('周黑马')):
            labels.append('周黑马')
        if bool(row.get('月黑马')):
            labels.append('月黑马')
        if bool(row.get('日掘地')):
            labels.append('日掘地')
        if bool(row.get('周掘地')):
            labels.append('周掘地')
        if bool(row.get('月掘地')):
            labels.append('月掘地')
        return "、".join(labels) if labels else "无"
    df['信号类型'] = df.apply(_build_signal_text, axis=1)
    
    # 更新列配置
    column_config.update({
        "日🐴": st.column_config.TextColumn("日🐴", width="small", help="日线黑马"),
        "周🐴": st.column_config.TextColumn("周🐴", width="small", help="周线黑马"),
        "月🐴": st.column_config.TextColumn("月🐴", width="small", help="月线黑马"),
        "日⛏️": st.column_config.TextColumn("日⛏️", width="small", help="日线掘地"),
        "周⛏️": st.column_config.TextColumn("周⛏️", width="small", help="周线掘地"),
        "月⛏️": st.column_config.TextColumn("月⛏️", width="small", help="月线掘地"),
        "信号类型": st.column_config.TextColumn("信号类型", width="large", help="明确标注触发的是哪类信号"),
    })
    
    # === 应用信号筛选 ===
    heima_filter = st.session_state.get('heima_filter', '全部')
    before_heima_count = len(df)
    
    # 统计
    day_heima_count = df['日黑马'].sum()
    week_heima_count = df['周黑马'].sum()
    month_heima_count = df['月黑马'].sum()
    day_juedi_count = df['日掘地'].sum()
    week_juedi_count = df['周掘地'].sum()
    month_juedi_count = df['月掘地'].sum()
    
    # 黑马筛选
    if heima_filter == "有日黑马":
        df = df[df['日黑马'] == True]
    elif heima_filter == "有周黑马":
        df = df[df['周黑马'] == True]
    elif heima_filter == "有月黑马":
        df = df[df['月黑马'] == True]
    elif heima_filter == "有任意黑马":
        df = df[(df['日黑马'] == True) | (df['周黑马'] == True) | (df['月黑马'] == True)]
    # 掘地筛选
    elif heima_filter == "有日掘地":
        df = df[df['日掘地'] == True]
    elif heima_filter == "有周掘地":
        df = df[df['周掘地'] == True]
    elif heima_filter == "有月掘地":
        df = df[df['月掘地'] == True]
    elif heima_filter == "有任意掘地":
        df = df[(df['日掘地'] == True) | (df['周掘地'] == True) | (df['月掘地'] == True)]

    # 显示列顺序
    display_cols = ['Rank_Score', '信号方向', 'mmoe_dir_prob', 'mmoe_return_5d', 'mmoe_return_20d', 'mmoe_max_dd', 'mmoe_score', '新发现', '信号类型', '日🐴', '周🐴', '月🐴', '日⛏️', '周⛏️', '月⛏️', '新闻', '大师建议', 'Ticker', 'Name', 'Mkt Cap', 'Cap_Category', '信号日期', '信号价', '现价', '价格变化(%)', 'Turnover', 'Day BLUE', 'Week BLUE', 'Month BLUE', 'ADX', 'Strategy', '筹码形态', 'Wave_Desc', 'Chan_Desc', 'Stop Loss', 'Shares Rec', 'Regime']
    existing_cols = [c for c in display_cols if c in df.columns]

    # === 按用户要求分4个标签页 ===
    # 预先计算各类别数据
    has_day = df['Day BLUE'] > 0 if 'Day BLUE' in df.columns else False
    has_week = df['Week BLUE'] > 0 if 'Week BLUE' in df.columns else False
    has_month = df['Month BLUE'] > 0 if 'Month BLUE' in df.columns else False
    
    # 1. 只日BLUE: Day > 0, Week = 0
    df_day_only = df[has_day & ~has_week] if 'Day BLUE' in df.columns else df.head(0)
    
    # 2. 日周/只周: (Day > 0 AND Week > 0) OR (Day = 0 AND Week > 0)
    df_day_week = df[(has_day & has_week) | (~has_day & has_week)] if 'Week BLUE' in df.columns else df.head(0)
    
    # 3. 日周月/只月: (Day > 0 AND Week > 0 AND Month > 0) OR (Month > 0)
    df_month = df[(has_day & has_week & has_month) | has_month] if 'Month BLUE' in df.columns else df.head(0)
    
    # 4. 特殊信号 (黑马/掘地) - 只要有黑马或掘地就显示，不管日周月
    heima_cache_key = f"heima_cache_{selected_date}_{selected_market}"
    if heima_cache_key in st.session_state:
        heima_data = st.session_state[heima_cache_key]
        df['黑马'] = df['Ticker'].map(lambda t: heima_data.get(t, {}).get('heima', False))
        df['掘地'] = df['Ticker'].map(lambda t: heima_data.get(t, {}).get('juedi', False))
        df_special = df[(df['黑马'] == True) | (df['掘地'] == True)].copy()
    else:
        df_special = df.head(0)
    
    # 计算各标签页数量
    count_day_only = len(df_day_only)
    count_day_week = len(df_day_week)
    count_month = len(df_month)
    count_special = len(df_special)
    
    # === 信号筛选状态 + 排序工具栏 ===
    st.markdown("---")
    toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([2, 1, 1])
    
    with toolbar_col1:
        if heima_filter != "全部":
            if len(df) == 0:
                st.warning(f"⚠️ **{heima_filter}**: 当天无符合条件的股票")
            else:
                st.success(f"✅ **{heima_filter}** (共 {len(df)} 只)")
        else:
            st.caption(f"🐴 黑马: 日{day_heima_count} 周{week_heima_count} 月{month_heima_count} | ⛏️ 掘地: 日{day_juedi_count} 周{week_juedi_count} 月{month_juedi_count}")
    
    with toolbar_col2:
        sort_options = {
            "综合评分": "Rank_Score",
            "日 BLUE": "Day BLUE",
            "周 BLUE": "Week BLUE",
            "月 BLUE": "Month BLUE",
            "ADX": "ADX",
            "成交额": "Turnover",
            "市值": "Mkt Cap",
            "价格": "Price"
        }
        available_sort = {k: v for k, v in sort_options.items() if v in df.columns}
        user_sort = st.selectbox("排序", list(available_sort.keys()), index=0, key="scan_sort_by",
                                 label_visibility="collapsed")
    
    with toolbar_col3:
        sort_asc = st.toggle("升序", value=False, key="scan_sort_asc")
    
    # 全局排序
    sort_col_name = available_sort.get(user_sort, "Day BLUE")
    if sort_col_name in df.columns:
        df = df.sort_values(sort_col_name, ascending=sort_asc, na_position='last')
    
    # 创建标签页 (增加板块热度)
    tab_day_only, tab_day_week, tab_month, tab_special, tab_sector = st.tabs([
        f"📈 只日线 ({count_day_only})",
        f"📊 日+周线 ({count_day_week})",
        f"📅 含月线 ({count_month})",
        f"🐴⛏️ 特殊信号 ({count_special})",
        "🔥 板块热度"
    ])
    
    # 用于存储各标签页选择的行 (用于深度透视)
    selected_ticker = None
    selected_row_data = None
    compare_tickers = []  # 多选时用于对比
    
    def _handle_table_selection(event, df_source):
        """处理表格选择事件，返回 (单选ticker, 单选row, 对比ticker列表)"""
        if event and hasattr(event, 'selection') and event.selection.rows:
            rows = event.selection.rows
            valid_rows = [r for r in rows if r < len(df_source)]
            if len(valid_rows) == 1:
                return df_source.iloc[valid_rows[0]]['Ticker'], df_source.iloc[valid_rows[0]], []
            elif len(valid_rows) > 1:
                tickers = [df_source.iloc[r]['Ticker'] for r in valid_rows[:4]]  # 最多4只
                return None, None, tickers
        return None, None, []
    
    with tab_day_only:
        st.caption("💡 只有日线信号，尚未形成周线共振，适合短线。选1行=详情，选多行=对比")
        if len(df_day_only) > 0:
            event1 = st.dataframe(
                df_day_only[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
                on_select="rerun",
                key="df_day_only"
            )
            t, r, c = _handle_table_selection(event1, df_day_only)
            if t: selected_ticker, selected_row_data = t, r
            if c: compare_tickers = c
        else:
            st.info("暂无只有日线信号的股票")
    
    with tab_day_week:
        st.caption("💡 日周双信号共振 或 周线独立信号，中期趋势确认。选1行=详情，选多行=对比")
        if len(df_day_week) > 0:
            event2 = st.dataframe(
                df_day_week[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
                on_select="rerun",
                key="df_day_week"
            )
            t, r, c = _handle_table_selection(event2, df_day_week)
            if t: selected_ticker, selected_row_data = t, r
            if c: compare_tickers = c
        else:
            st.info("暂无日周共振或周线信号的股票")
    
    with tab_month:
        st.caption("💡 日周月三重共振 或 月线信号，大级别底部机会。选1行=详情，选多行=对比")
        if len(df_month) > 0:
            event3 = st.dataframe(
                df_month[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
                on_select="rerun",
                key="df_month"
            )
            t, r, c = _handle_table_selection(event3, df_month)
            if t: selected_ticker, selected_row_data = t, r
            if c: compare_tickers = c
        else:
            st.info("暂无含月线信号的股票")
    
    with tab_special:
        st.caption("🐴 黑马 / ⛏️ 掘地 / 🔥 顶格峰：特殊形态信号")
        
        # === 扫描范围选择 ===
        scan_scope = st.radio(
            "扫描范围",
            options=["📋 当前信号股", "🌐 全量股票"],
            horizontal=True,
            help="当前信号股=只扫描已有BLUE信号的股票 | 全量股票=扫描市场所有股票",
            key="special_scan_scope"
        )
        
        # 根据选择确定扫描列表
        if scan_scope == "📋 当前信号股":
            scan_tickers = df['Ticker'].tolist()
            scope_label = "当前信号股"
        else:
            # 全量扫描 - 从 Polygon API 获取所有股票
            try:
                from data_fetcher import get_all_us_tickers, get_all_cn_tickers
                if selected_market == 'CN':
                    scan_tickers = get_all_cn_tickers()
                else:
                    scan_tickers = get_all_us_tickers()
                # 限制数量，避免太慢
                if len(scan_tickers) > 3000:
                    scan_tickers = scan_tickers[:3000]
                scope_label = f"全量扫描 ({len(scan_tickers)} 只)"
            except Exception as e:
                scan_tickers = df['Ticker'].tolist()
                scope_label = f"当前信号股 (全量失败: {e})"
        
        st.caption(f"📊 扫描范围: {scope_label} | 共 {len(scan_tickers)} 只股票")
        
        # === 特殊信号缓存 ===
        special_cache_key = f"special_signals_{selected_date}_{selected_market}_{scan_scope}"
        
        if special_cache_key not in st.session_state:
            st.info("需要扫描特殊信号（含幻影主力），点击下方按钮开始")
            
            if st.button("🔍 扫描全部信号 (黑马+幻影+金叉+背离)", key="scan_special", type="primary"):
                from concurrent.futures import ThreadPoolExecutor, as_completed
                from indicator_utils import calculate_heima_full, calculate_phantom_indicator, calculate_adx_series
                from chart_utils import quick_chip_analysis
                
                results = {}
                
                def calc_special_signals(ticker):
                    """计算单只股票的全部特殊信号"""
                    base = {'heima': False, 'juedi': False, 'bottom_peak': False,
                            'phantom_escape': False, 'phantom_buy': False, 'phantom_pink': 50,
                            'golden_bottom': False, 'top_divergence': False,
                            'triple_escape': False, 'two_golden_cross': False,
                            'cci': 0}
                    try:
                        stock_df = fetch_data_from_polygon(ticker, days=250)
                        if stock_df is None or len(stock_df) < 30:
                            return ticker, base
                        
                        h = stock_df['High'].values
                        l = stock_df['Low'].values
                        c = stock_df['Close'].values
                        o = stock_df['Open'].values
                        v = stock_df['Volume'].values
                        
                        # 完整黑马 (含金叉、背离等)
                        if len(stock_df) >= 50:
                            hf = calculate_heima_full(h, l, c, o, v)
                            base['heima'] = bool(hf['heima'][-1])
                            base['juedi'] = bool(hf['juedi'][-1])
                            base['golden_bottom'] = bool(hf['golden_bottom'][-1])
                            base['top_divergence'] = bool(hf['top_divergence'][-1])
                            base['two_golden_cross'] = bool(hf['two_golden_cross'][-1])
                            base['cci'] = float(hf['CCI'][-1])
                        
                        # 顶格峰信号
                        try:
                            chip = quick_chip_analysis(stock_df)
                            if chip and (chip.get('is_strong_bottom_peak') or chip.get('is_bottom_peak')):
                                base['bottom_peak'] = True
                        except:
                            pass
                        
                        # 幻影主力
                        if len(stock_df) >= 50:
                            try:
                                ph = calculate_phantom_indicator(o, h, l, c, v)
                                adx_arr = calculate_adx_series(h, l, c)
                                adx_v = float(adx_arr[-1])
                                base['phantom_pink'] = float(ph['pink'][-1])
                                
                                is_sell = bool(ph['sell_signal'][-1])
                                green_v = float(ph['green'][-1])
                                
                                # 三重逃顶: 顶背离 + PINK>80 + 资金流出
                                if base['top_divergence'] and ph['pink'][-1] > 80 and green_v < 0:
                                    base['triple_escape'] = True
                                
                                # 幻影逃顶: PINK下穿90 + 资金流出
                                if is_sell and green_v < 0 and adx_v < 30:
                                    base['phantom_escape'] = True
                                
                                # 趋势回调: BLUE消失 + ADX>25
                                is_blue_dis = bool(ph['blue_disappear'][-1])
                                if is_blue_dis and adx_v >= 25:
                                    base['phantom_buy'] = True
                            except:
                                pass
                        
                        return ticker, base
                    except:
                        return ticker, base
                
                progress = st.progress(0, text="正在扫描特殊信号...")
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(calc_special_signals, t): t for t in scan_tickers}
                    completed = 0
                    for future in as_completed(futures):
                        ticker, signals = future.result()
                        results[ticker] = signals
                        completed += 1
                        progress.progress(completed / len(scan_tickers), text=f"扫描中 {completed}/{len(scan_tickers)}")
                
                progress.empty()
                st.session_state[special_cache_key] = results
                
                # 统计结果
                heima_count = sum(1 for r in results.values() if r['heima'])
                juedi_count = sum(1 for r in results.values() if r['juedi'])
                peak_count = sum(1 for r in results.values() if r['bottom_peak'])
                escape_count = sum(1 for r in results.values() if r['phantom_escape'])
                pbuy_count = sum(1 for r in results.values() if r['phantom_buy'])
                gb_count = sum(1 for r in results.values() if r['golden_bottom'])
                te_count = sum(1 for r in results.values() if r['triple_escape'])
                td_count = sum(1 for r in results.values() if r['top_divergence'])
                st.success(f"✅ 扫描完成！🎯 黄金底: {gb_count} | 🚨 三重逃顶: {te_count} | 🐴 黑马: {heima_count} | ⛏️ 掘地: {juedi_count} | 🔥 顶格峰: {peak_count} | ⚠️ 顶背离: {td_count}")
                st.rerun()
        else:
            # 显示结果
            signal_data = st.session_state[special_cache_key]
            
            # 信号过滤器
            all_filter_opts = [
                "🎯 黄金底", "🚨 三重逃顶", "⚠️ 顶背离", "⚡ 二次金叉",
                "🐴 黑马", "⛏️ 掘地", "🔥 顶格峰",
                "📈 趋势回调", "🔴 幻影逃顶"
            ]
            filter_opts = st.multiselect(
                "筛选信号类型",
                all_filter_opts,
                default=all_filter_opts,
                key="special_filter"
            )
            
            # 构建特殊信号数据框
            special_rows = []
            for ticker, signals in signal_data.items():
                has_any = (
                    (signals.get('golden_bottom') and "🎯 黄金底" in filter_opts) or
                    (signals.get('triple_escape') and "🚨 三重逃顶" in filter_opts) or
                    (signals.get('top_divergence') and "⚠️ 顶背离" in filter_opts) or
                    (signals.get('two_golden_cross') and "⚡ 二次金叉" in filter_opts) or
                    (signals['heima'] and "🐴 黑马" in filter_opts) or
                    (signals['juedi'] and "⛏️ 掘地" in filter_opts) or
                    (signals['bottom_peak'] and "🔥 顶格峰" in filter_opts) or
                    (signals['phantom_escape'] and "🔴 幻影逃顶" in filter_opts) or
                    (signals['phantom_buy'] and "📈 趋势回调" in filter_opts)
                )
                if has_any:
                    signal_types = []
                    if signals.get('golden_bottom') and "🎯 黄金底" in filter_opts:
                        signal_types.append('🎯黄金底')
                    if signals.get('triple_escape') and "🚨 三重逃顶" in filter_opts:
                        signal_types.append('🚨三重逃顶')
                    if signals.get('top_divergence') and "⚠️ 顶背离" in filter_opts:
                        signal_types.append('⚠️顶背离')
                    if signals.get('two_golden_cross') and "⚡ 二次金叉" in filter_opts:
                        signal_types.append('⚡二次金叉')
                    if signals['heima'] and "🐴 黑马" in filter_opts:
                        signal_types.append('🐴黑马')
                    if signals['juedi'] and "⛏️ 掘地" in filter_opts:
                        signal_types.append('⛏️掘地')
                    if signals['bottom_peak'] and "🔥 顶格峰" in filter_opts:
                        signal_types.append('🔥顶格峰')
                    if signals['phantom_escape'] and "🔴 幻影逃顶" in filter_opts:
                        signal_types.append('🔴逃顶')
                    if signals['phantom_buy'] and "📈 趋势回调" in filter_opts:
                        signal_types.append('📈回调')
                    
                    if not signal_types:
                        continue
                    
                    # 尝试从 df 获取更多信息
                    ticker_info = df[df['Ticker'] == ticker]
                    if len(ticker_info) > 0:
                        row = ticker_info.iloc[0].to_dict()
                        row['信号类型'] = ' '.join(signal_types)
                        row['PINK'] = round(signals.get('phantom_pink', 50), 1)
                        row['CCI'] = round(signals.get('cci', 0), 0)
                        special_rows.append(row)
                    else:
                        special_rows.append({
                            'Ticker': ticker,
                            '信号类型': ' '.join(signal_types),
                            'PINK': round(signals.get('phantom_pink', 50), 1),
                            'CCI': round(signals.get('cci', 0), 0)
                        })
            
            if special_rows:
                df_special_result = pd.DataFrame(special_rows)
                
                # 统计显示
                st.markdown(f"**找到 {len(special_rows)} 只特殊信号股票**")
                
                display_with_signal = ['信号类型', 'PINK', 'CCI'] + existing_cols
                cols_to_show = [c for c in display_with_signal if c in df_special_result.columns]
                
                event4 = st.dataframe(
                    df_special_result[cols_to_show],
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    key="df_special"
                )
                if event4 and hasattr(event4, 'selection') and event4.selection.rows:
                    idx = event4.selection.rows[0]
                    if idx < len(df_special_result):
                        selected_ticker = df_special_result.iloc[idx]['Ticker']
                        selected_row_data = df_special_result.iloc[idx]
            else:
                st.info("暂无匹配的特殊信号股票")
            
            # 清除缓存按钮
            if st.button("🔄 重新扫描", key="rescan_special"):
                del st.session_state[special_cache_key]
                st.rerun()

    # === 板块热度标签页 ===
    with tab_sector:
        st.caption("🔥 行业板块涨跌幅排名 - 追踪市场热点")
        
        from data_fetcher import get_sector_data, get_cn_sector_data_period, get_us_sector_data_period
        
        # 分析模式选择
        analysis_mode = st.radio(
            "分析模式",
            options=["📊 基础模式", "🔥 增强模式"],
            horizontal=True,
            key="sector_analysis_mode",
            help="增强模式显示量比、连涨天数、资金流向、综合热度"
        )
        
        if analysis_mode == "🔥 增强模式":
            # 增强模式：显示热度评分
            from data_fetcher import get_cn_sector_enhanced, get_us_sector_enhanced
            
            enhanced_key = f"sector_enhanced_{selected_market}"
            
            if st.button("🔄 刷新增强数据", key="refresh_enhanced"):
                if enhanced_key in st.session_state:
                    del st.session_state[enhanced_key]
            
            if enhanced_key not in st.session_state:
                with st.spinner("正在计算增强指标..."):
                    try:
                        if selected_market == 'CN':
                            enhanced_df = get_cn_sector_enhanced()
                        else:
                            enhanced_df = get_us_sector_enhanced()
                        st.session_state[enhanced_key] = enhanced_df
                    except Exception as e:
                        st.error(f"获取增强数据失败: {e}")
                        enhanced_df = None
            
            enhanced_df = st.session_state.get(enhanced_key)
            
            if enhanced_df is not None and len(enhanced_df) > 0:
                st.markdown("### 🔥 板块热度排行 (综合评分)")
                st.caption("评分 = 涨幅(30%) + 量比(25%) + 连涨(25%) + 资金流(20%)")
                
                # 格式化显示
                display_df = enhanced_df.copy()
                display_df['change_pct'] = display_df['change_pct'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.2f}x")
                display_df['consecutive_days'] = display_df['consecutive_days'].apply(lambda x: f"{x}天" if x > 0 else "-")
                if 'money_flow' in display_df.columns:
                    display_df['money_flow'] = display_df['money_flow'].apply(lambda x: f"+{x:.1f}亿" if x > 0 else f"{x:.1f}亿")
                display_df['heat_score'] = display_df['heat_score'].apply(lambda x: f"🔥{x:.0f}" if x >= 50 else f"{x:.0f}")
                
                display_cols = ['name', 'change_pct', 'volume_ratio', 'consecutive_days', 'heat_score']
                if 'money_flow' in display_df.columns:
                    display_cols.insert(4, 'money_flow')
                
                st.dataframe(
                    display_df[display_cols],
                    column_config={
                        'name': '板块',
                        'change_pct': '涨跌幅',
                        'volume_ratio': '量比',
                        'consecutive_days': '连涨',
                        'money_flow': '资金流',
                        'heat_score': '热度'
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # 可视化热度前10
                if len(enhanced_df) >= 5:
                    import plotly.express as px
                    top10 = enhanced_df.head(10)
                    fig = px.bar(
                        top10, x='name', y='heat_score',
                        title="🔥 热度 Top 10 板块",
                        color='heat_score',
                        color_continuous_scale='YlOrRd'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暂无增强数据")
        
        else:
            # 基础模式：原有逻辑
            # 时间段选择
            period_options = {
                "📅 今日": "1d",
                "📆 本周": "1w", 
                "📊 本月": "1m",
                "📈 今年": "ytd"
            }
            selected_period_label = st.radio(
                "时间范围",
                options=list(period_options.keys()),
                horizontal=True,
                key="sector_period"
            )
            selected_period = period_options[selected_period_label]
            
            # 缓存板块数据 (按时间段)
            sector_cache_key = f"sector_data_{selected_market}_{selected_period}"
            
            col_refresh, col_info = st.columns([1, 3])
            with col_refresh:
                if st.button("🔄 刷新", key="refresh_sector"):
                    # 清除所有时间段缓存
                    for p in period_options.values():
                        key = f"sector_data_{selected_market}_{p}"
                        if key in st.session_state:
                            del st.session_state[key]
            
            if sector_cache_key not in st.session_state:
                with st.spinner(f"正在获取{selected_period_label}板块数据..."):
                    try:
                        if selected_market == 'CN':
                            sector_df = get_cn_sector_data_period(period=selected_period)
                        else:
                            sector_df = get_us_sector_data_period(period=selected_period)
                        
                        if sector_df is not None:
                            st.session_state[sector_cache_key] = sector_df
                    except Exception as e:
                        st.error(f"获取数据失败: {e}")
                        sector_df = None
        
            if sector_cache_key in st.session_state:
                sector_df = st.session_state[sector_cache_key]
                
                if sector_df is not None and len(sector_df) > 0:
                    # 统计信息
                    up_count = len(sector_df[sector_df['change_pct'] > 0])
                    down_count = len(sector_df[sector_df['change_pct'] < 0])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("板块数量", len(sector_df))
                    with col2:
                        st.metric("🔴 上涨", up_count)
                    with col3:
                        st.metric("🟢 下跌", down_count)
                    
                    st.divider()
                    
                    # 分两列显示：涨幅榜和跌幅榜
                    col_up, col_down = st.columns(2)
                    
                    with col_up:
                        st.markdown(f"### 📈 {selected_period_label} 涨幅榜 Top 15")
                        top_up = sector_df.head(15).copy()
                        top_up['change_pct'] = top_up['change_pct'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                        if 'amount' in top_up.columns:
                            top_up['amount'] = top_up['amount'].apply(lambda x: f"{x:.1f}亿" if pd.notna(x) else "N/A")
                        if 'stock_count' in top_up.columns:
                            display_cols_up = ['name', 'change_pct', 'amount', 'stock_count']
                        else:
                            display_cols_up = ['name', 'change_pct']
                        cols_to_show = [c for c in display_cols_up if c in top_up.columns]
                        st.dataframe(
                            top_up[cols_to_show],
                            column_config={
                                'name': '板块',
                                'change_pct': '涨跌幅',
                                'amount': '成交额',
                                'stock_count': '股票数'
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    with col_down:
                        st.markdown(f"### 📉 {selected_period_label} 跌幅榜 Top 15")
                        top_down = sector_df.tail(15).iloc[::-1].copy()
                        top_down['change_pct'] = top_down['change_pct'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                        if 'amount' in top_down.columns:
                            top_down['amount'] = top_down['amount'].apply(lambda x: f"{x:.1f}亿" if pd.notna(x) else "N/A")
                        if 'stock_count' in top_down.columns:
                            display_cols_down = ['name', 'change_pct', 'amount', 'stock_count']
                        else:
                            display_cols_down = ['name', 'change_pct']
                        cols_to_show = [c for c in display_cols_down if c in top_down.columns]
                        st.dataframe(
                            top_down[cols_to_show],
                            column_config={
                                'name': '板块',
                                'change_pct': '涨跌幅',
                                'amount': '成交额',
                                'stock_count': '股票数'
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    st.info("暂无板块数据")
                
                # === 板块详情区域 ===
                st.divider()
                st.markdown("### 🔍 板块详情")
                
                # 板块选择下拉框
                sector_names = sector_df['name'].tolist()
                selected_sector = st.selectbox(
                    "选择板块查看详情",
                    options=sector_names,
                    key="sector_detail_select"
                )
                
                if selected_sector:
                    with st.expander(f"📊 {selected_sector} 详情", expanded=True):
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown("#### 🔥 板块热门股")
                            
                            # 获取该板块的热门股票
                            hot_stocks_key = f"hot_stocks_{selected_sector}_{selected_market}"
                            
                            if hot_stocks_key not in st.session_state:
                                with st.spinner("加载热门股..."):
                                    try:
                                        if selected_market == 'CN':
                                            from data_fetcher import get_cn_sector_hot_stocks
                                            hot_df = get_cn_sector_hot_stocks(selected_sector)
                                        else:
                                            from data_fetcher import get_us_sector_hot_stocks
                                            hot_df = get_us_sector_hot_stocks(selected_sector)
                                        st.session_state[hot_stocks_key] = hot_df
                                    except Exception as e:
                                        st.session_state[hot_stocks_key] = None
                            
                            hot_df = st.session_state.get(hot_stocks_key)
                            if hot_df is not None and len(hot_df) > 0:
                                st.dataframe(
                                    hot_df[['name', 'pct_chg']].head(10),
                                    column_config={
                                        'name': '股票',
                                        'pct_chg': '涨跌幅%'
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                            else:
                                st.info("暂无热门股数据")
                        
                        with detail_col2:
                            st.markdown("#### 📰 相关新闻")
                            
                            # 显示新闻搜索链接
                            if selected_market == 'CN':
                                search_term = f"{selected_sector}板块 股票 新闻"
                                baidu_url = f"https://www.baidu.com/s?wd={search_term}"
                                st.markdown(f"🔗 [百度搜索: {selected_sector}新闻]({baidu_url})")
                                
                                eastmoney_url = f"https://so.eastmoney.com/news/s?keyword={selected_sector}"
                                st.markdown(f"🔗 [东方财富: {selected_sector}]({eastmoney_url})")
                            else:
                                search_term = f"{selected_sector} sector stocks news"
                                google_url = f"https://www.google.com/search?q={search_term}&tbm=nws"
                                st.markdown(f"🔗 [Google News: {selected_sector}]({google_url})")
                                
                                yahoo_url = f"https://finance.yahoo.com/quote/{sector_df[sector_df['name']==selected_sector]['sector'].values[0] if len(sector_df[sector_df['name']==selected_sector]) > 0 else 'XLK'}"
                                st.markdown(f"🔗 [Yahoo Finance]({yahoo_url})")
                            
                            st.caption("💡 点击链接查看最新市场资讯")
            else:
                st.info("正在加载板块数据...")


    # === 收集所有选中的股票 (用于批量分析) ===
    selected_tickers_set = set()
    
    # 辅助函数: 安全获取 event
    def collect_from_event(evt, source_df):
        if evt and hasattr(evt, 'selection') and evt.selection.rows:
            return [source_df.iloc[i]['Ticker'] for i in evt.selection.rows if i < len(source_df)]
        return []

    if 'event1' in locals(): selected_tickers_set.update(collect_from_event(event1, df_day_only))
    if 'event2' in locals(): selected_tickers_set.update(collect_from_event(event2, df_day_week))
    if 'event3' in locals(): selected_tickers_set.update(collect_from_event(event3, df_month))
    
    # === 🚀 批量深度分析工作台 ===
    if len(selected_tickers_set) > 0:
        st.divider()
        st.subheader(f"🚀 深度分析工作台 (已选 {len(selected_tickers_set)} 只)")
        
        selected_list = list(selected_tickers_set)
        
        # 批量分析按钮
        col_act, col_info = st.columns([1, 4])
        with col_act:
            do_batch_analyze = st.button("✨ 分析选中股票", type="primary", use_container_width=True)
            
        with col_info:
            st.caption(f"选中: {', '.join(selected_list[:10])} {'...' if len(selected_list)>10 else ''}")

        if do_batch_analyze:
            with st.status("正在进行全方位深度扫描...", expanded=True) as status:
                try:
                    from strategies.master_strategies import analyze_stock_for_master, get_master_summary_for_stock
                    if selected_market == 'US':
                        from data_fetcher import get_us_stock_data as get_data
                    else:
                        from data_fetcher import get_cn_stock_data as get_data

                    # 获取缓存
                    master_cache_key = f"master_analysis_{selected_date}_{selected_market}"
                    master_details_key = f"{master_cache_key}_details"
                    
                    if master_cache_key not in st.session_state: st.session_state[master_cache_key] = {}
                    if master_details_key not in st.session_state: st.session_state[master_details_key] = {}
                    
                    master_res = st.session_state[master_cache_key]
                    master_details = st.session_state[master_details_key]
                    
                    prog_bar = st.progress(0)
                    for i, ticker in enumerate(selected_list):
                        status.write(f"正在分析 {ticker}...")
                        try:
                            # 1. 获取近期历史数据
                            hist_df = get_data(ticker, days=40)
                            
                            if hist_df is not None and not hist_df.empty:
                                # 准备参数
                                current_row = df[df['Ticker'] == ticker].iloc[0]
                                price = float(current_row.get('Price', 0))
                                sma5 = hist_df['Close'].rolling(5).mean().iloc[-1]
                                sma20 = hist_df['Close'].rolling(20).mean().iloc[-1]
                                vol = hist_df['Volume'].iloc[-1]
                                vol_ma5 = hist_df['Volume'].rolling(5).mean().iloc[-1]
                                vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0
                                
                                # 简易 TD
                                c = hist_df['Close'].values
                                td_count = 0
                                if len(c) > 13:
                                    if c[-1] > c[-5]: # 上涨
                                        count = 0
                                        for k in range(1, 10):
                                            if c[-k] > c[-k-4]: count += 1
                                            else: break
                                        td_count = count
                                
                                # 调用大师分析
                                analyses = analyze_stock_for_master(
                                    symbol=ticker,
                                    blue_daily=float(current_row.get('Day BLUE', 0)),
                                    blue_weekly=float(current_row.get('Week BLUE', 0)),
                                    blue_monthly=float(current_row.get('Month BLUE', 0)),
                                    adx=float(current_row.get('ADX', 0)),
                                    vol_ratio=vol_ratio,
                                    change_pct=float(hist_df['Close'].pct_change().iloc[-1] * 100),
                                    price=price,
                                    sma5=sma5,
                                    sma20=sma20,
                                    td_count=td_count,
                                    is_heima=True if '黑马' in str(current_row.get('Strategy', '')) else False
                                )
                                
                                # 汇总
                                summary = get_master_summary_for_stock(
                                    analyses,
                                    profile=st.session_state.get("master_profile", "medium")
                                )
                                master_res[ticker] = summary
                                master_details[ticker] = analyses
                                
                        except Exception as e:
                            print(f"Error analyzing {ticker}: {e}")
                        
                        prog_bar.progress((i + 1) / len(selected_list))
                    
                    # 更新缓存
                    st.session_state[master_cache_key] = master_res
                    st.session_state[master_details_key] = master_details
                    
                    # 重新计算 Rank
                    from ml.ranking_system import get_ranking_system
                    ranker = get_ranking_system()
                    # 这里不需要重新计算整个 df，只需要展示部分
                    
                    st.success("✅ 分析完成！请查看下方 Alpha Picks 报告")
                    st.session_state['show_batch_results'] = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"批量分析失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # === 展示批量分析结果 (Alpha Picks 风格) ===
        if st.session_state.get('show_batch_results', False):
            st.markdown("### 👑 选中股票优选报告 (Alpha Picks)")
            
            # 临时计算这些股票的 Rank Score
            try:
                from ml.ranking_system import get_ranking_system
                ranker = get_ranking_system()
                master_res = st.session_state.get(f"master_analysis_{selected_date}_{selected_market}", {})
                
                # 只对选中的股票计算
                subset_df = df[df['Ticker'].isin(selected_list)].copy()
                scored_df = ranker.calculate_integrated_score(subset_df, master_results=master_res)
                
                # 展示前 5 名
                top_picks = scored_df.head(5)
                
                cols = st.columns(len(top_picks))
                for i, (_, row) in enumerate(top_picks.iterrows()):
                    with cols[i]:
                        score = row['Rank_Score']
                        ticker = row['Ticker']
                        
                        tags = []
                        if score >= 80: tags.append("🔥 强推")
                        if master_res.get(ticker): tags.append("🤖 大师")
                        
                        with st.container(border=True):
                            name = row.get('Name', '')
                            # 如果名称缺失，尝试补充获取
                            if pd.isna(name) or str(name).strip() == '' or str(name) == 'nan':
                                try:
                                    from data_fetcher import get_cn_ticker_details, get_ticker_details
                                    if selected_market == 'CN':
                                        info_dict = get_cn_ticker_details(ticker)
                                    else:
                                        info_dict = get_ticker_details(ticker)
                                    
                                    if info_dict and info_dict.get('name'):
                                        name = info_dict.get('name')
                                except:
                                    name = ticker

                            st.metric(f"{ticker}", f"{score:.0f}分", str(name)[:6])
                            st.progress(score/100)
                            st.caption(" ".join(tags))
                            
                            if st.button(f"详情", key=f"btn_detail_{ticker}"):
                                selected_ticker = ticker
                                selected_row_data = row
                                st.rerun() # 触发下方深度透视
                                
            except Exception as e:
                st.error(f"结果展示出错: {e}")
            st.divider()

    # 4. 深度透视 / 对比模式
    if compare_tickers and len(compare_tickers) >= 2:
        st.divider()
        st.markdown(f"### ⚖️ 股票对比 ({len(compare_tickers)} 只)")
        _render_stock_comparison(compare_tickers, selected_market, f"scan_{selected_date}")
        st.warning("⚠️ **免责声明**: 以上仅为量化模型生成的参考信号，不构成投资建议。请结合大盘环境自主决策。")
    elif selected_ticker is not None and selected_row_data is not None:
        st.divider()
        
        # 使用统一的股票详情组件
        render_unified_stock_detail(
            symbol=selected_ticker,
            market=selected_market,
            key_prefix=f"scan_{selected_date}"
        )
        
        st.warning("⚠️ **免责声明**: 以上仅为量化模型生成的参考信号，不构成投资建议。请结合大盘环境自主决策。")
    else:
        st.info("👈 点击1行=详情分析 | 选中多行=并排对比")

    # === 旧代码已被统一组件替代 (render_unified_stock_detail) ===
    # 原有功能包括: 全面智能诊断、大师分析、舆情分析、筹码分析等
    # 全部整合进 components/stock_detail.py，如需查看原实现请查看 git 历史
    
    # 删除旧代码占位符 - 开始删除标记
    _LEGACY_CODE_REMOVED = True  # 以下到 "删除标记结束" 之间的代码已删除
    # 旧代码 (670+行) 已删除，请查看 git 历史
    # 删除范围: 原 AI 诊断、大师分析、舆情分析、K线图表、筹码分析等
    # 替代方案: 全部功能已整合到 render_unified_stock_detail 组件
    
    # === 删除旧代码开始标记 ===
    if False:  # 永不执行 - 保留结构以便未来参考
        # 原有代码包括:
        # - AI 综合诊断 (LLMAnalyzer.generate_decision_dashboard)
        # - 大师量化视角 (master_strategies.analyze_stock_for_master)
        # - 社区舆情分析 (social_monitor.get_social_report)
        # - K线图表 (create_candlestick_chart_dynamic)
        # - 筹码分析 (analyze_chip_flow)
        # - BLUE 信号 (calculate_blue_signal_series)
        # 全部功能已迁移至 components/stock_detail.py
        pass
    # === 旧代码已删除 - 全部功能已迁移至 render_unified_stock_detail ===
    # 
    # 以下代码块 (原约670行) 已被删除:
    # - AI 诊断与决策仪表盘
    # - 大师量化分析 (蔡森/TD/萧明道/黑马/BLUE)
    # - 社区舆情监控
    # - K线图表 (日/周/月线)
    # - 筹码分布与主力动向分析
    # - 技术指标展示
    # - 风控与仓位建议
    #
    # 替代方案: 全部功能已整合到 components/stock_detail.py
    # 查看原实现请使用: git show HEAD~1:versions/v3/app.py
    #
    # === 保留大师分析详情查看器 (删除旧代码但保留此功能) ===

    # === 大师分析详情查看器 (全局) ===
    master_details_key = f"master_analysis_{selected_date}_{selected_market}_details"
    
    if master_details_key in st.session_state:
        st.divider()
        st.header("🔍 大师分析实验室 (Master's Lab)")
        
        details = st.session_state[master_details_key]
        analyzed_tickers = list(details.keys())
        
        if analyzed_tickers:
            col_sel, col_content = st.columns([1, 3])
            
            with col_sel:
                # 尝试获取股票名称
                def get_stock_label(tk):
                    name = ""
                    if 'Ticker' in df.columns and 'Name' in df.columns:
                        matches = df[df['Ticker'] == tk]
                        if not matches.empty:
                            name = matches['Name'].iloc[0]
                    return f"{tk} {name}"
                
                selected_ticker_for_detail = st.radio(
                    "已分析股票", 
                    analyzed_tickers,
                    format_func=get_stock_label
                )
            
            with col_content:
                if selected_ticker_for_detail:
                    analyses = details[selected_ticker_for_detail]
                    
                    # 1. 总体评价
                    from strategies.master_strategies import get_master_summary_for_stock
                    summary = get_master_summary_for_stock(
                        analyses,
                        profile=st.session_state.get("master_profile", "medium")
                    )
                    
                    # 显示共识信号
                    signal = summary.get('overall_signal', 'HOLD')
                    if signal == 'BUY':
                        st.success(f"### {summary['overall_action']}")
                    elif signal == 'SELL':
                        st.error(f"### {summary['overall_action']}")
                    elif signal == 'CONFLICT':
                        st.warning(f"### {summary['overall_action']}")
                    else:
                        st.info(f"### {summary['overall_action']}")
                    
                    # 显示指标
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🟢 看多", f"{summary['buy_votes']} (权重:{summary.get('weighted_buy', 0):.1f})")
                    c2.metric("🔴 看空", f"{summary['sell_votes']} (权重:{summary.get('weighted_sell', 0):.1f})")
                    c3.metric("⚪ 观望", summary['hold_votes'])
                    c4.metric("📊 共识度", f"{summary.get('consensus_score', 0):.0f}%")
                    
                    # 冲突警告
                    if summary.get('conflict_warning'):
                        st.warning(summary['conflict_warning'])
                    
                    if summary['best_opportunity']:
                        st.info(f"**最佳机会**: {summary['best_opportunity']}")
                    if summary['key_risk']:
                        st.warning(f"**主要风险**: {summary['key_risk']}")
                    
                    st.divider()
                    
                    # 2. 各大师详细观点
                    for key, analysis in analyses.items():
                        with st.expander(f"{analysis.icon} {analysis.master}: {analysis.action_emoji} {analysis.action}", expanded=True):
                            st.markdown(f"**判断逻辑**: {analysis.reason}")
                            st.markdown(f"**操作建议**: {analysis.operation}")
                            
                            if analysis.stop_loss:
                                st.markdown(f"🛑 **止损**: {analysis.stop_loss}")
                            if analysis.take_profit:
                                st.markdown(f"🎯 **目标**: {analysis.take_profit}")
                            
                            st.caption(f"信心指数: {'⭐' * analysis.confidence}")

    elif analyze_master: # 如果还没有详情但按钮被按了 (状态中)
        pass # 等待上面rerun
    else:
        st.divider()
        st.caption("ℹ️ 点击上方的 '🤖 大师深度分析' 按钮，可在此处查看 5 位大师对前 20 只股票的详细会诊报告。")



def render_stock_lookup_page():
    """个股深度分析 - 输入任意代码获取完整分析 (扫描数据+排名+ML预测+图表+新闻)"""
    st.header("🔍 个股深度分析")
    
    # --- 搜索栏 ---
    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
    with search_col1:
        symbol_input = st.text_input(
            "股票代码", value=st.session_state.get('lookup_symbol', ''), 
            placeholder="AAPL, NVDA, 600519, 000001...",
            key="lookup_input"
        )
        symbol = symbol_input.upper().strip() if symbol_input else ""
    
    with search_col2:
        # 自动检测市场
        is_cn = symbol and (symbol.isdigit() and len(symbol) == 6) or symbol.endswith(('.SH', '.SZ'))
        market_options = {"🇺🇸 美股": "US", "🇨🇳 A股": "CN"}
        default_idx = 1 if is_cn else 0
        lookup_market = st.radio("市场", options=list(market_options.keys()), 
                                 index=default_idx, horizontal=True, key="lookup_market")
        market = market_options[lookup_market]
        _set_active_market(market)
    
    with search_col3:
        search_btn = st.button("🔍 开始分析", type="primary", use_container_width=True)
        # 热门快捷
        hot_col1, hot_col2 = st.columns(2)
        with hot_col1:
            if st.button("NVDA", key="hot_nvda", use_container_width=True):
                st.session_state['lookup_symbol'] = 'NVDA'
                st.rerun()
        with hot_col2:
            if st.button("AAPL", key="hot_aapl", use_container_width=True):
                st.session_state['lookup_symbol'] = 'AAPL'
                st.rerun()
    
    # 保存搜索
    if search_btn and symbol:
        st.session_state['lookup_symbol'] = symbol
    
    # --- 主分析区 ---
    active_symbol = st.session_state.get('lookup_symbol', '')
    if not active_symbol:
        st.info("👆 输入股票代码并点击「开始分析」，获取完整的个股深度报告。")
        
        # 最近搜索历史
        search_history = st.session_state.get('lookup_history', [])
        if search_history:
            st.markdown("**最近搜索:**")
            hist_cols = st.columns(min(len(search_history), 6))
            for i, h in enumerate(search_history[:6]):
                with hist_cols[i]:
                    if st.button(h, key=f"hist_{h}", use_container_width=True):
                        st.session_state['lookup_symbol'] = h
                        st.rerun()
        return
    
    # 更新搜索历史
    history = st.session_state.get('lookup_history', [])
    if active_symbol not in history:
        history.insert(0, active_symbol)
        st.session_state['lookup_history'] = history[:10]
    
    st.divider()
    
    # --- 1. 扫描数据概览 (如果该股票在最近扫描中) ---
    scan_info = _get_scan_info_for_symbol(active_symbol, market)
    
    if scan_info:
        st.markdown("### 📋 扫描数据")
        info_cols = st.columns(7)
        with info_cols[0]:
            st.metric("扫描日期", scan_info.get('scan_date', 'N/A'))
        with info_cols[1]:
            blue_d = scan_info.get('blue_daily', 0) or 0
            st.metric("日BLUE", f"{float(blue_d):.0f}")
        with info_cols[2]:
            blue_w = scan_info.get('blue_weekly', 0) or 0
            st.metric("周BLUE", f"{float(blue_w):.0f}")
        with info_cols[3]:
            blue_m = scan_info.get('blue_monthly', 0) or 0
            st.metric("月BLUE", f"{float(blue_m):.0f}")
        with info_cols[4]:
            adx = scan_info.get('adx', 0) or 0
            st.metric("ADX", f"{float(adx):.1f}")
        with info_cols[5]:
            rank = scan_info.get('rank_score', 0) or 0
            st.metric("综合评分", f"{float(rank):.1f}")
        with info_cols[6]:
            signals = []
            if scan_info.get('heima_daily'): signals.append("🐴日")
            if scan_info.get('heima_weekly'): signals.append("🐴周")
            if scan_info.get('heima_monthly'): signals.append("🐴月")
            if scan_info.get('juedi_daily'): signals.append("⛏️日")
            if scan_info.get('juedi_weekly'): signals.append("⛏️周")
            if scan_info.get('juedi_monthly'): signals.append("⛏️月")
            st.metric("特殊信号", " ".join(signals) if signals else "无")
        
        # 策略标签
        strategy = scan_info.get('strategy', '')
        if strategy:
            st.caption(f"策略: {strategy} | 市值: {scan_info.get('cap_category', 'N/A')} | 波动: {scan_info.get('regime', 'N/A')}")
        st.divider()
    else:
        st.caption(f"ℹ️ {active_symbol} 不在最近的扫描结果中 (非信号股或未被扫描)")

    # --- 1.5 Qlib 融合解释层 ---
    try:
        from pathlib import Path
        qlib_dir = Path(current_dir) / "ml" / "saved_models" / f"qlib_{market.lower()}"
        seg_path = qlib_dir / "segment_strategy_compare_latest.csv"
        if seg_path.exists():
            seg_df = pd.read_csv(seg_path)
            if not seg_df.empty and "best_sharpe" in seg_df.columns:
                top_seg = seg_df.sort_values("best_sharpe", ascending=False).iloc[0]
                best_segment = str(top_seg.get("segment", "")).upper()

                cap_cat = ""
                if scan_info:
                    cap_cat = str(scan_info.get("cap_category", "") or "")
                in_focus = False
                if best_segment == "LARGE":
                    in_focus = ("Large" in cap_cat) or ("Mega" in cap_cat)
                elif best_segment == "MID":
                    in_focus = "Mid" in cap_cat
                elif best_segment == "SMALL":
                    in_focus = ("Small" in cap_cat) or ("Micro" in cap_cat)

                if in_focus:
                    st.success(
                        f"🧠 Qlib 融合判断: 当前更偏好 **{best_segment}** 分层，"
                        f"{active_symbol} 与该分层匹配。"
                    )
                else:
                    st.info(
                        f"🧠 Qlib 融合判断: 当前更偏好 **{best_segment}** 分层，"
                        f"而 {active_symbol} 的市值分层匹配度一般。"
                    )
    except Exception:
        pass

    # --- 1.6 多维决策评分卡（Qlib 只是其中一维） ---
    def _clip_score(x):
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(0.0, min(100.0, v))

    # 维度1: 技术趋势 (BLUE/ADX)
    blue_d = float((scan_info or {}).get("blue_daily", 0) or 0)
    adx_v = float((scan_info or {}).get("adx", 0) or 0)
    score_tech = _clip_score(blue_d * 0.55 + min(adx_v, 50) * 0.9)

    # 维度2: 信号质量 (黑马/掘地/周月共振)
    sig_cnt = 0
    for k in ["heima_daily", "heima_weekly", "heima_monthly", "juedi_daily", "juedi_weekly", "juedi_monthly"]:
        if (scan_info or {}).get(k):
            sig_cnt += 1
    score_signal = _clip_score(sig_cnt * 16 + (10 if float((scan_info or {}).get("blue_weekly", 0) or 0) > 100 else 0))

    # 维度3: 流动性与可执行性
    turnover_v = float((scan_info or {}).get("turnover_m", (scan_info or {}).get("turnover", 0)) or 0)
    price_v = float((scan_info or {}).get("price", 0) or 0)
    score_liquidity = _clip_score(min(turnover_v, 30) / 30 * 80 + (20 if price_v >= 1 else 0))

    # 维度4: 风险收益结构（使用系统综合分近似）
    rank_v = float((scan_info or {}).get("rank_score", 0) or 0)
    score_risk_reward = _clip_score(rank_v)

    # 维度5: Qlib 维度（分层匹配度）
    qlib_dim = 50.0
    try:
        from pathlib import Path
        qlib_dir = Path(current_dir) / "ml" / "saved_models" / f"qlib_{market.lower()}"
        seg_path = qlib_dir / "segment_strategy_compare_latest.csv"
        if seg_path.exists():
            seg_df = pd.read_csv(seg_path)
            if not seg_df.empty and "best_sharpe" in seg_df.columns:
                top_seg = seg_df.sort_values("best_sharpe", ascending=False).iloc[0]
                best_segment = str(top_seg.get("segment", "")).upper()
                cap_cat = str((scan_info or {}).get("cap_category", "") or "")
                if best_segment == "LARGE":
                    qlib_dim = 85 if ("Large" in cap_cat or "Mega" in cap_cat) else 45
                elif best_segment == "MID":
                    qlib_dim = 85 if ("Mid" in cap_cat) else 45
                elif best_segment == "SMALL":
                    qlib_dim = 85 if ("Small" in cap_cat or "Micro" in cap_cat) else 45
    except Exception:
        pass

    dim_df = pd.DataFrame(
        [
            {"维度": "技术趋势", "分数": round(score_tech, 1), "说明": "BLUE + ADX"},
            {"维度": "信号质量", "分数": round(score_signal, 1), "说明": "黑马/掘地/共振"},
            {"维度": "流动性", "分数": round(score_liquidity, 1), "说明": "成交额 + 价格"},
            {"维度": "风险收益", "分数": round(score_risk_reward, 1), "说明": "综合评分近似"},
            {"维度": "Qlib维度", "分数": round(qlib_dim, 1), "说明": "分层匹配度"},
        ]
    )
    total_score = round(
        score_tech * 0.30
        + score_signal * 0.20
        + score_liquidity * 0.15
        + score_risk_reward * 0.20
        + qlib_dim * 0.15,
        1,
    )
    st.markdown("### 🧭 多维决策评分卡")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(dim_df, use_container_width=True, hide_index=True)
    with c2:
        st.metric("综合决策分", f"{total_score}")
        st.caption("Qlib 仅占 15% 权重")
    
    # --- 2. 历史信号轨迹 ---
    _render_signal_history(active_symbol, market)
    
    # --- 3. 统一详情组件 (图表/筹码/指标/AI/ML/新闻/操作) ---
    render_unified_stock_detail(
        symbol=active_symbol,
        market=market,
        key_prefix=f"lookup_{active_symbol}"
    )
    
    st.warning("⚠️ **免责声明**: 以上仅为量化模型生成的参考信号，不构成投资建议。")


def _get_scan_info_for_symbol(symbol: str, market: str) -> dict:
    """获取某只股票在最近扫描中的数据"""
    try:
        dates = _cached_scanned_dates(market=market)
        if not dates:
            return {}
        
        # 查最近3天的扫描结果
        for d in dates[:3]:
            results = _cached_scan_results(scan_date=d, market=market, limit=1000)
            for r in results:
                if r.get('symbol', '').upper() == symbol.upper():
                    r['scan_date'] = d
                    return r
        return {}
    except:
        return {}


def _render_signal_history(symbol: str, market: str):
    """渲染该股票的历史信号轨迹 (最近30天)"""
    try:
        dates = _cached_scanned_dates(market=market)
        if not dates or len(dates) < 2:
            return
        
        history_data = []
        for d in dates[:30]:
            results = _cached_scan_results(scan_date=d, market=market, limit=1000)
            for r in results:
                if r.get('symbol', '').upper() == symbol.upper():
                    history_data.append({
                        '日期': d,
                        '日BLUE': float(r.get('blue_daily', 0) or 0),
                        '周BLUE': float(r.get('blue_weekly', 0) or 0),
                        '月BLUE': float(r.get('blue_monthly', 0) or 0),
                        'ADX': float(r.get('adx', 0) or 0),
                        '评分': float(r.get('rank_score', 0) or 0),
                    })
                    break
        
        if not history_data:
            return
        
        with st.expander(f"📈 历史信号轨迹 ({len(history_data)} 天)", expanded=False):
            hist_df = pd.DataFrame(history_data)
            
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df['日期'], y=hist_df['日BLUE'], name='日BLUE', 
                                     line=dict(color='#58a6ff', width=2)))
            fig.add_trace(go.Scatter(x=hist_df['日期'], y=hist_df['周BLUE'], name='周BLUE', 
                                     line=dict(color='#a371f7', width=2)))
            fig.add_trace(go.Scatter(x=hist_df['日期'], y=hist_df['月BLUE'], name='月BLUE', 
                                     line=dict(color='#3fb950', width=2)))
            fig.add_trace(go.Scatter(x=hist_df['日期'], y=hist_df['ADX'], name='ADX', 
                                     line=dict(color='#d29922', width=1, dash='dot')))
            fig.update_layout(
                height=250, margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="", yaxis_title="信号强度",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"signal_hist_{symbol}")
            
            # 出现天数统计
            total_days = len(dates[:30])
            signal_days = len(history_data)
            st.caption(f"最近 {total_days} 个交易日中，{symbol} 出现在扫描信号中 **{signal_days} 天** ({signal_days/total_days*100:.0f}%)")
    except Exception as e:
        pass  # 静默失败




def render_signal_tracker_page():
    """信号追踪页面 - 查看历史扫描信号的后续表现"""
    st.header("📈 信号追踪 (Signal Tracker)")
    
    # 导入数据库函数
    from db.database import (
        get_signal_history, get_portfolio, add_to_watchlist, 
        add_trade, get_trades, update_watchlist_status, delete_from_watchlist
    )
    
    # Tab 结构 - 新增"今日信号"
    tab0, tab1, tab2, tab3 = st.tabs(["🎯 今日信号", "📊 信号表现", "🔍 信号复盘", "💼 我的持仓"])
    
    # ==================== Tab 0: 今日买卖信号 (新增) ====================
    with tab0:
        st.info("🔔 每日买入/卖出信号推荐")
        render_todays_signals_tab()
    
    # ==================== Tab 1: 信号表现 (原有功能) ====================
    with tab1:
        st.info("查看历史扫描结果中股票的后续走势，验证信号有效性。")
        render_signal_performance_tab()
    
    # ==================== Tab 2: 信号复盘 ====================
    with tab2:
        st.info("选择股票查看历史所有信号点及后续表现")
        render_signal_review_tab()
    
    # ==================== Tab 3: 我的持仓 ====================
    with tab3:
        st.info("添加并跟踪你的实际持仓，记录交易")
        render_portfolio_tab()


def render_todays_signals_tab():
    """今日买卖信号 Tab"""
    # 侧边栏设置
    with st.sidebar:
        st.subheader("🎯 信号设置")
        
        market = st.radio(
            "选择市场",
            ["🇺🇸 美股", "🇨🇳 A股"],
            horizontal=True,
            key="signal_market"
        )
        market_code = "US" if "美股" in market else "CN"
        
        min_confidence = st.slider("最低信心度", 30, 90, 50, key="signal_conf")
        
        generate_btn = st.button("🔄 生成今日信号", type="primary", use_container_width=True)
    
    # 尝试导入信号系统
    try:
        from strategies.signal_system import get_signal_manager, SignalType
        manager = get_signal_manager()
    except Exception as e:
        st.error(f"信号系统加载失败: {e}")
        return
    
    # 生成信号
    if generate_btn:
        with st.spinner("正在生成交易信号..."):
            result = manager.generate_daily_signals(market=market_code)
            
            if 'error' in result:
                st.error(f"生成失败: {result['error']}")
            else:
                st.success(f"✅ 生成 {result.get('buy_signals', 0)} 个买入信号, {result.get('sell_signals', 0)} 个卖出信号")
                st.rerun()
    
    # 显示今日信号
    todays_signals = manager.get_todays_signals(market=market_code)
    
    if not todays_signals:
        st.warning("暂无今日信号，点击「生成今日信号」按钮")
        
        # 显示说明
        st.markdown("""
        ### 📋 信号类型说明
        
        | 信号 | 说明 | 操作建议 |
        |------|------|----------|
        | 🟢 **买入** | 满足买入条件 | 考虑建仓 |
        | 🔴 **卖出** | 获利回吐或趋势转弱 | 减仓或清仓 |
        | 🛑 **止损** | 跌破止损位 | 立即止损 |
        | 🎯 **止盈** | 达到目标价 | 落袋为安 |
        | 👀 **观察** | 待确认信号 | 继续观察 |
        
        ### 💡 信号强度
        
        - 🔥 **强烈**: 多条件共振，信心 > 70%
        - ⚡ **中等**: 主要条件满足，信心 50-70%
        - 💧 **弱**: 单一条件触发，信心 < 50%
        """)
        return
    
    # 过滤低信心度信号
    todays_signals = [s for s in todays_signals if s.get('confidence', 0) >= min_confidence]
    
    # 分类显示
    buy_signals = [s for s in todays_signals if s['signal_type'] == '买入']
    sell_signals = [s for s in todays_signals if s['signal_type'] in ['卖出', '止损', '止盈']]
    
    # 买入信号
    st.subheader("🟢 买入信号")
    if buy_signals:
        buy_df = pd.DataFrame([{
            '代码': s['symbol'],
            '强度': s['strength'],
            '价格': f"${s['price']:.2f}" if market_code == 'US' else f"¥{s['price']:.2f}",
            '目标': f"${s['target_price']:.2f}" if market_code == 'US' else f"¥{s['target_price']:.2f}",
            '止损': f"${s['stop_loss']:.2f}" if market_code == 'US' else f"¥{s['stop_loss']:.2f}",
            '策略': s['strategy'],
            '信心': f"{s['confidence']:.0f}%",
            '理由': s['reason']
        } for s in buy_signals])
        
        st.dataframe(buy_df, hide_index=True, use_container_width=True)
        
        # 可视化
        if len(buy_signals) > 0:
            st.markdown("#### 📊 信心度分布")
            chart_data = pd.DataFrame({
                '股票': [s['symbol'] for s in buy_signals[:10]],
                '信心度': [s['confidence'] for s in buy_signals[:10]]
            })
            st.bar_chart(chart_data.set_index('股票'), height=200)
    else:
        st.info("暂无买入信号")
    
    st.divider()
    
    # 卖出信号
    st.subheader("🔴 卖出/止损信号")
    if sell_signals:
        sell_df = pd.DataFrame([{
            '代码': s['symbol'],
            '类型': s['signal_type'],
            '强度': s['strength'],
            '价格': f"${s['price']:.2f}" if market_code == 'US' else f"¥{s['price']:.2f}",
            '策略': s['strategy'],
            '信心': f"{s['confidence']:.0f}%",
            '理由': s['reason']
        } for s in sell_signals])
        
        st.dataframe(sell_df, hide_index=True, use_container_width=True)
    else:
        st.info("暂无卖出信号")
    
    st.divider()
    
    # 历史信号统计
    st.subheader("📈 近7日信号统计")
    
    historical = manager.get_historical_signals(days=7, market=market_code)
    if historical:
        # 按日期统计
        date_counts = {}
        for s in historical:
            date = s.get('generated_at', 'Unknown')
            if date not in date_counts:
                date_counts[date] = {'买入': 0, '卖出': 0}
            if s['signal_type'] == '买入':
                date_counts[date]['买入'] += 1
            else:
                date_counts[date]['卖出'] += 1
        
        if date_counts:
            stats_df = pd.DataFrame([
                {'日期': date, '买入信号': counts['买入'], '卖出信号': counts['卖出']}
                for date, counts in date_counts.items()
            ])
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    else:
        st.info("暂无历史信号数据")


def render_signal_performance_tab():
    """信号表现 Tab (原有功能)"""
    from services.signal_tracker_service import batch_calculate_returns

    # 侧边栏设置
    with st.sidebar:
        st.subheader("📊 追踪设置")
        
        # 市场选择
        market = st.radio(
            "选择市场",
            ["🇺🇸 美股", "🇨🇳 A股"],
            horizontal=True,
            key="tracker_market"
        )
        market_code = "US" if "美股" in market else "CN"
        
        # 获取历史扫描日期
        dates = _cached_scanned_dates(market=market_code)
        
        if not dates:
            st.warning(f"暂无 {market} 的历史扫描数据")
            return
        
        # 日期选择
        selected_date = st.selectbox(
            "选择扫描日期",
            options=dates[:30],  # 最近30天
            index=0,
            help="选择要追踪的历史扫描日期"
        )
        
        # 追踪天数
        track_days = st.slider("追踪天数", 5, 30, 20)
        
        # 计算按钮
        calculate_btn = st.button("🔍 计算信号表现", type="primary", use_container_width=True)
    
    # 主区域
    if not calculate_btn:
        # 显示说明
        st.markdown("""
        ### 使用说明
        
        1. 在左侧选择 **市场** 和 **历史扫描日期**
        2. 点击 **"计算信号表现"** 按钮
        3. 系统将分析该日期扫描出的信号在后续的表现
        
        #### 指标说明
        - **胜率**: 信号后续上涨的比例
        - **平均收益**: 所有信号的平均收益率
        - **5D/10D/20D**: 信号后 5/10/20 个交易日的收益
        """)
        
        # 显示可用日期概览
        if dates:
            st.markdown("### 📅 可用历史日期")
            
            # 获取每个日期的信号数量
            date_info = []
            for d in dates[:10]:
                count = len(_cached_scan_results(scan_date=d, market=market_code, limit=1000))
                date_info.append({'日期': d, '信号数': count})
            
            if date_info:
                st.dataframe(pd.DataFrame(date_info), hide_index=True, use_container_width=True)
        return
    
    # 执行计算
    with st.spinner(f"正在计算 {selected_date} 的信号表现..."):
        # 获取该天的扫描结果
        scan_results = _cached_scan_results(scan_date=selected_date, market=market_code, limit=100)
        
        if not scan_results:
            st.error("该日期没有扫描结果")
            return
        
        st.success(f"找到 {len(scan_results)} 个信号，正在计算后续表现...")
        
        # 准备信号列表
        signals = [{
            'symbol': r['symbol'],
            'signal_date': selected_date,
            'day_blue': r.get('blue_daily', 0),
            'week_blue': r.get('blue_weekly', 0),
            'name': r.get('name', ''),
            'entry_price': r.get('price', 0)
        } for r in scan_results]
        
        # 批量计算收益
        progress_bar = st.progress(0, text="计算中...")
        returns = batch_calculate_returns(signals, market_code, max_workers=15)
        progress_bar.progress(100, text="计算完成!")
        
        if not returns:
            st.warning("无法获取足够的历史数据来计算收益")
            return
    
    # 转换为 DataFrame
    df = pd.DataFrame(returns)
    
    # 显示统计摘要
    st.markdown("---")
    st.markdown("### 📊 整体表现统计")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = len(returns)
        st.metric("分析信号数", f"{total}")
    
    with col2:
        if 'return_5d' in df.columns:
            valid_5d = df['return_5d'].dropna()
            avg_5d = valid_5d.mean() if len(valid_5d) > 0 else 0
            st.metric("平均 5D 收益", f"{avg_5d:+.2f}%",
                     delta="盈利" if avg_5d > 0 else "亏损",
                     delta_color="normal" if avg_5d > 0 else "inverse")
    
    with col3:
        if 'return_10d' in df.columns:
            valid_10d = df['return_10d'].dropna()
            avg_10d = valid_10d.mean() if len(valid_10d) > 0 else 0
            st.metric("平均 10D 收益", f"{avg_10d:+.2f}%",
                     delta="盈利" if avg_10d > 0 else "亏损",
                     delta_color="normal" if avg_10d > 0 else "inverse")
    
    with col4:
        if 'return_20d' in df.columns:
            valid_20d = df['return_20d'].dropna()
            avg_20d = valid_20d.mean() if len(valid_20d) > 0 else 0
            st.metric("平均 20D 收益", f"{avg_20d:+.2f}%",
                     delta="盈利" if avg_20d > 0 else "亏损",
                     delta_color="normal" if avg_20d > 0 else "inverse")
    
    with col5:
        if 'return_20d' in df.columns:
            valid = df['return_20d'].dropna()
            if len(valid) > 0:
                win_rate = len(valid[valid > 0]) / len(valid) * 100
                st.metric("20D 胜率", f"{win_rate:.0f}%",
                         delta="优秀" if win_rate > 60 else ("一般" if win_rate > 40 else "较差"))
    
    # 信号分类
    st.markdown("### 🎯 信号分类")
    
    if 'return_20d' in df.columns:
        df_valid = df.dropna(subset=['return_20d'])
        
        excellent = df_valid[df_valid['return_20d'] > 10]
        good = df_valid[(df_valid['return_20d'] > 0) & (df_valid['return_20d'] <= 10)]
        poor = df_valid[df_valid['return_20d'] <= 0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"### ✅ 优质信号: {len(excellent)}")
            st.caption("20D 收益 > 10%")
            if len(excellent) > 0:
                st.write(f"平均 BLUE: {excellent['day_blue'].mean():.0f}")
        
        with col2:
            st.info(f"### 🟡 一般信号: {len(good)}")
            st.caption("20D 收益 0-10%")
            if len(good) > 0:
                st.write(f"平均 BLUE: {good['day_blue'].mean():.0f}")
        
        with col3:
            st.warning(f"### ❌ 差信号: {len(poor)}")
            st.caption("20D 收益 < 0%")
            if len(poor) > 0:
                st.write(f"平均 BLUE: {poor['day_blue'].mean():.0f}")
    
    # 详细数据表格
    st.markdown("### 📋 详细数据")
    
    # 准备显示数据
    display_df = df[['symbol', 'name', 'day_blue', 'entry_price', 
                     'return_5d', 'return_10d', 'return_20d', 
                     'max_gain', 'max_drawdown', 'current_return']].copy()
    
    display_df.columns = ['代码', '名称', 'Day BLUE', '入场价', 
                          '5D收益', '10D收益', '20D收益', 
                          '最大涨幅', '最大回撤', '当前收益']
    
    # 格式化
    for col in ['5D收益', '10D收益', '20D收益', '最大涨幅', '最大回撤', '当前收益']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )
    
    display_df['入场价'] = display_df['入场价'].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) and x > 0 else "N/A"
    )
    
    # 排序选项
    sort_col = st.selectbox("排序方式", ['20D收益', '10D收益', '5D收益', 'Day BLUE'], key="sort_col")
    
    # 因为已经格式化为字符串，需要对原始数据排序
    sort_map = {'20D收益': 'return_20d', '10D收益': 'return_10d', '5D收益': 'return_5d', 'Day BLUE': 'day_blue'}
    if sort_map[sort_col] in df.columns:
        sort_idx = df[sort_map[sort_col]].sort_values(ascending=False).index
        display_df = display_df.loc[sort_idx]
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # 信号质量评估
    st.markdown("### 💡 信号质量评估")
    
    if 'return_20d' in df.columns:
        valid_20d = df['return_20d'].dropna()
        if len(valid_20d) > 0:
            avg_return = valid_20d.mean()
            win_rate = len(valid_20d[valid_20d > 0]) / len(valid_20d) * 100
            
            if avg_return > 5 and win_rate > 55:
                st.success(f"""
                **✅ 优质信号批次**
                
                - 平均 20D 收益: {avg_return:.2f}%
                - 胜率: {win_rate:.0f}%
                - 优质信号占比: {len(excellent)/len(df_valid)*100:.0f}%
                
                该批次信号表现优秀，策略参数有效！
                """)
            elif avg_return > 0 and win_rate > 40:
                st.info(f"""
                **🟡 一般信号批次**
                
                - 平均 20D 收益: {avg_return:.2f}%
                - 胜率: {win_rate:.0f}%
                
                该批次信号表现一般，建议结合其他指标筛选。
                """)
            else:
                st.warning(f"""
                **⚠️ 低质量信号批次**
                
                - 平均 20D 收益: {avg_return:.2f}%
                - 胜率: {win_rate:.0f}%
                
                该批次信号表现不佳，建议调整策略参数。
                """)


def render_signal_review_tab():
    """信号复盘 Tab - 查看个股历史信号"""
    from db.database import get_signal_history
    
    st.markdown("### 🔍 选择股票查看历史信号")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("输入股票代码", value="", placeholder="例如: NVDA, AAPL", key="review_symbol")
    
    with col2:
        market = st.radio("市场", ["US", "CN"], horizontal=True, key="review_market")
    
    if not symbol:
        st.info("请输入股票代码开始查看信号历史")
        return
    
    symbol = symbol.upper().strip()
    
    with st.spinner(f"正在加载 {symbol} 的历史信号..."):
        signals = get_signal_history(symbol, market=market, limit=100)
    
    if not signals:
        st.warning(f"未找到 {symbol} 的历史信号记录")
        return
    
    st.success(f"找到 {len(signals)} 条历史信号记录")
    
    # 转换为 DataFrame
    df = pd.DataFrame(signals)
    
    # 显示信号列表
    st.markdown("### 📋 历史信号列表")
    
    display_cols = ['scan_date', 'price', 'blue_daily', 'blue_weekly', 'wave_phase']
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()
    display_df.columns = ['信号日期', '当日价格', 'Day BLUE', 'Week BLUE', '波浪阶段'][:len(available_cols)]
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # 信号统计
    st.markdown("### 📊 信号统计")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("信号次数", len(signals))
    with col2:
        avg_blue = df['blue_daily'].mean() if 'blue_daily' in df.columns else 0
        st.metric("平均 Day BLUE", f"{avg_blue:.0f}" if avg_blue else "N/A")
    with col3:
        max_blue = df['blue_daily'].max() if 'blue_daily' in df.columns else 0
        st.metric("最高 Day BLUE", f"{max_blue:.0f}" if max_blue else "N/A")


def render_portfolio_tab():
    """我的持仓 Tab - 实盘持仓 + 模拟交易"""
    from db.database import (
        get_portfolio, add_to_watchlist, add_trade, 
        get_trades, update_watchlist_status, delete_from_watchlist
    )
    from services.portfolio_service import (
        get_portfolio_summary, calculate_portfolio_pnl,
        get_paper_account, paper_buy, paper_sell, 
        get_paper_trades, reset_paper_account,
        get_paper_equity_curve, get_paper_monthly_returns, get_realized_pnl_history,
        list_paper_accounts, create_paper_account,
        get_paper_account_config, update_paper_account_config
    )
    
    # 选择模式
    mode = st.radio(
        "选择模式",
        ["💼 实盘持仓", "🎮 模拟交易"],
        horizontal=True,
        key="portfolio_mode"
    )
    
    st.divider()
    
    # ==================== 实盘持仓模式 ====================
    if mode == "💼 实盘持仓":
        # 权限检查
        if not is_admin():
            st.warning("⚠️ 持仓管理需要管理员权限，您当前为访客模式（只读）")
            st.markdown("---")
        
        # 获取持仓汇总
        with st.spinner("正在获取实时数据..."):
            summary = get_portfolio_summary()
        
        # 汇总统计卡片
        if summary['positions'] > 0:
            st.subheader("📊 持仓汇总")
            
            m1, m2, m3, m4 = st.columns(4)
            
            pnl_color = "normal" if summary['total_pnl'] >= 0 else "inverse"
            
            m1.metric("总成本", f"${summary['total_cost']:,.2f}")
            m2.metric("总市值", f"${summary['total_market_value']:,.2f}")
            m3.metric("未实现盈亏", f"${summary['total_pnl']:+,.2f}", 
                     f"{summary['total_pnl_pct']:+.2f}%", delta_color=pnl_color)
            m4.metric("持仓数", f"{summary['positions']} 只",
                     f"🟢{summary['winners']} 🔴{summary['losers']}")
            
            st.divider()
        
        # 添加持仓表单 (仅管理员可见)
        if is_admin():
            with st.expander("➕ 添加新持仓", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    new_symbol = st.text_input("股票代码", placeholder="NVDA", key="add_symbol")
                with col2:
                    new_price = st.number_input("买入价格", min_value=0.01, value=100.0, key="add_price")
                with col3:
                    new_shares = st.number_input("股数", min_value=1, value=100, key="add_shares")
                
                col4, col5 = st.columns(2)
                with col4:
                    new_market = st.selectbox("市场", ["US", "CN"], key="add_market")
                with col5:
                    new_date = st.date_input("买入日期", key="add_date")
                
                notes = st.text_input("备注", placeholder="可选", key="add_notes")
                
                if st.button("✅ 添加持仓", type="primary"):
                    if new_symbol:
                        symbol = new_symbol.upper().strip()
                        entry_date = new_date.strftime('%Y-%m-%d')
                        
                        add_to_watchlist(symbol, new_price, new_shares, entry_date, new_market, 'holding', notes)
                        add_trade(symbol, 'BUY', new_price, new_shares, entry_date, new_market, notes)
                        
                        st.success(f"✅ 已添加 {symbol} 到持仓")
                        st.rerun()
                    else:
                        st.error("请输入股票代码")
        
        # 当前持仓列表 (带实时盈亏)
        st.subheader("💼 当前持仓")
        
        if summary.get('details'):
            for item in summary['details']:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{item['symbol']}**")
                        st.caption(f"成本: ${item['entry_price']:.2f} × {item['shares']}")
                    
                    with col2:
                        if item.get('current_price'):
                            st.markdown(f"现价: **${item['current_price']:.2f}**")
                            st.caption(f"市值: ${item['market_value']:,.2f}")
                        else:
                            st.markdown("现价: --")
                    
                    with col3:
                        if item.get('unrealized_pnl') is not None:
                            pnl = item['unrealized_pnl']
                            pnl_pct = item['unrealized_pnl_pct']
                            color = "green" if pnl >= 0 else "red"
                            st.markdown(f"盈亏: <span style='color:{color}'>${pnl:+,.2f}</span>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"<span style='color:{color}'>{pnl_pct:+.2f}%</span>", 
                                       unsafe_allow_html=True)
                        else:
                            st.markdown("盈亏: --")
                    
                    with col4:
                        st.caption(f"买入: {item['entry_date']}")
                        st.caption(f"市场: {item['market']}")
                    
                    with col5:
                        if is_admin():
                            if st.button("卖出", key=f"sell_{item['id']}"):
                                st.session_state[f"show_sell_{item['id']}"] = True
                    
                    # 卖出对话框
                    if is_admin() and st.session_state.get(f"show_sell_{item['id']}"):
                        sell_price = st.number_input(
                            f"卖出价格", 
                            min_value=0.01, 
                            value=float(item.get('current_price') or item['entry_price']),
                            key=f"sell_price_{item['id']}"
                        )
                        if st.button(f"确认卖出 {item['symbol']}", key=f"confirm_sell_{item['id']}"):
                            add_trade(item['symbol'], 'SELL', sell_price, item['shares'], 
                                     datetime.now().strftime('%Y-%m-%d'), item['market'])
                            update_watchlist_status(item['symbol'], item['entry_date'], 'sold', item['market'])
                            st.success(f"✅ 已卖出 {item['symbol']}")
                            st.session_state[f"show_sell_{item['id']}"] = False
                            st.rerun()
                
                st.divider()
        else:
            st.info("暂无持仓，点击上方添加")
        
        # 交易历史
        with st.expander("📜 交易历史", expanded=False):
            trades = get_trades(limit=20)
            if trades:
                df = pd.DataFrame(trades)
                display_df = df[['symbol', 'trade_type', 'price', 'shares', 'trade_date', 'market']].copy()
                display_df.columns = ['代码', '类型', '价格', '股数', '日期', '市场']
                display_df['类型'] = display_df['类型'].map({'BUY': '🟢买入', 'SELL': '🔴卖出'})
                display_df['价格'] = display_df['价格'].apply(lambda x: f"${x:.2f}")
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.info("暂无交易记录")
    
    # ==================== 模拟交易模式 ====================
    else:
        st.subheader("🎮 模拟交易账户")
        st.caption("使用虚拟资金测试交易策略，不用真金白银")

        # 策略子账户选择
        sub_accounts = list_paper_accounts()
        sub_names = [a['account_name'] for a in sub_accounts] if sub_accounts else ['default']
        default_account = _get_global_paper_account_name()
        default_index = sub_names.index(default_account) if default_account in sub_names else 0
        selected_account = st.selectbox(
            "策略子账户",
            sub_names,
            index=default_index,
            key="portfolio_paper_account_name"
        )
        _set_global_paper_account_name(selected_account)
        account_cfg = get_paper_account_config(selected_account)
        st.info(f"当前查看子账户: **{selected_account}**")
        st.caption(
            f"当前风控: 单票≤{float(account_cfg.get('max_single_position_pct', 0.30))*100:.1f}% | "
            f"最大回撤≤{float(account_cfg.get('max_drawdown_pct', 0.20))*100:.1f}%"
        )

        # 子账户总览（减少“我现在操作的是哪个账户”的歧义）
        with st.expander("👥 子账户总览", expanded=False):
            overview_rows = []
            for acc in sub_names:
                acc_data = get_paper_account(acc)
                if not acc_data:
                    continue
                overview_rows.append({
                    "账户": acc,
                    "现金": float(acc_data.get("cash_balance", 0) or 0),
                    "持仓市值": float(acc_data.get("position_value", 0) or 0),
                    "总权益": float(acc_data.get("total_equity", 0) or 0),
                    "持仓数": len(acc_data.get("positions", [])),
                })
            if overview_rows:
                ov_df = pd.DataFrame(overview_rows)
                for col in ["现金", "持仓市值", "总权益"]:
                    ov_df[col] = ov_df[col].apply(lambda x: f"${x:,.2f}")
                st.dataframe(ov_df, hide_index=True, use_container_width=True)
            else:
                st.caption("暂无子账户数据")

        with st.expander("➕ 新建策略子账户", expanded=False):
            new_sub_name = st.text_input("子账户名称", placeholder="trend_us / meanrev_cn", key="new_paper_subaccount")
            new_sub_cap = st.number_input("初始资金", min_value=1000.0, value=20000.0, step=1000.0, key="new_paper_sub_cap")
            if st.button("创建子账户", key="create_paper_subaccount_btn"):
                created = create_paper_account(new_sub_name.strip(), float(new_sub_cap))
                if created.get('success'):
                    st.success(f"✅ 子账户已创建: {created['account_name']}")
                    st.rerun()
                else:
                    st.error(f"❌ {created.get('error', '创建失败')}")

        with st.expander("🛡️ 子账户风控设置", expanded=False):
            strategy_note = st.text_area(
                "策略说明",
                value=account_cfg.get('strategy_note', ''),
                height=80,
                key="paper_account_strategy_note"
            )
            single_pos_limit = st.slider(
                "单票仓位上限 (%)",
                min_value=5,
                max_value=80,
                value=int(round(float(account_cfg.get('max_single_position_pct', 0.30)) * 100)),
                key="paper_account_single_pos_limit"
            )
            max_dd_limit = st.slider(
                "最大回撤阈值 (%)",
                min_value=5,
                max_value=80,
                value=int(round(float(account_cfg.get('max_drawdown_pct', 0.20)) * 100)),
                key="paper_account_max_dd_limit"
            )
            if st.button("保存风控设置", key="save_paper_account_config_btn"):
                saved = update_paper_account_config(
                    selected_account,
                    strategy_note=strategy_note,
                    max_single_position_pct=single_pos_limit / 100.0,
                    max_drawdown_pct=max_dd_limit / 100.0
                )
                if saved.get('success'):
                    st.success("✅ 已保存")
                    st.rerun()
                else:
                    st.error(f"❌ {saved.get('error', '保存失败')}")
        
        # 获取模拟账户
        with st.spinner("加载模拟账户..."):
            account = get_paper_account(selected_account)
        
        if not account:
            st.error("模拟账户加载失败")
            return
        
        # 账户汇总
        m1, m2, m3, m4 = st.columns(4)
        
        pnl_color = "normal" if account['total_pnl'] >= 0 else "inverse"
        
        m1.metric("初始资金", f"${account['initial_capital']:,.2f}")
        m2.metric("现金余额", f"${account['cash_balance']:,.2f}")
        m3.metric("持仓市值", f"${account['position_value']:,.2f}")
        m4.metric("总权益", f"${account['total_equity']:,.2f}",
                 f"{account['total_pnl_pct']:+.2f}%", delta_color=pnl_color)
        
        st.divider()
        
        # 交易面板
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            st.markdown("#### 🟢 买入")
            buy_symbol = st.text_input("股票代码", placeholder="AAPL", key="paper_buy_symbol")
            buy_shares = st.number_input("买入股数", min_value=1, value=10, key="paper_buy_shares")
            buy_price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0, key="paper_buy_price")
            buy_market = st.selectbox("市场", ["US", "CN"], key="paper_buy_market")
            buy_target_account = st.selectbox(
                "目标子账户",
                sub_names,
                index=sub_names.index(selected_account) if selected_account in sub_names else 0,
                key="paper_buy_target_account",
                help="买入会记入此子账户"
            )
            st.caption(f"本次买入将进入: **{buy_target_account}**")
            
            if st.button("🛒 买入", type="primary", key="do_paper_buy"):
                if buy_symbol:
                    price = buy_price if buy_price > 0 else None
                    result = paper_buy(buy_symbol.upper(), buy_shares, price, buy_market, buy_target_account)
                    
                    if result['success']:
                        st.success(
                            f"✅ 买入成功! {result['symbol']} {result['shares']}股 @ ${result['price']:.2f} "
                            f"→ 子账户[{buy_target_account}]"
                        )
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
                else:
                    st.error("请输入股票代码")
        
        with col_sell:
            st.markdown("#### 🔴 卖出")
            st.caption(f"当前卖出来源账户: **{selected_account}**")
            
            # 持仓下拉选择
            position_options = [f"{p['symbol']} ({p['shares']}股)" for p in account['positions']]
            if position_options:
                selected_pos = st.selectbox("选择持仓", position_options, key="paper_sell_select")
                sell_symbol = selected_pos.split(" ")[0] if selected_pos else ""
                
                # 找到选中的持仓
                selected_position = next((p for p in account['positions'] if p['symbol'] == sell_symbol), None)
                
                if selected_position:
                    max_shares = selected_position['shares']
                    sell_shares = st.number_input("卖出股数", min_value=1, max_value=max_shares, value=max_shares, key="paper_sell_shares")
                    sell_price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0, key="paper_sell_price")
                    
                    if st.button("💰 卖出", type="secondary", key="do_paper_sell"):
                        price = sell_price if sell_price > 0 else None
                        result = paper_sell(sell_symbol, sell_shares, price, selected_position['market'], selected_account)
                        
                        if result['success']:
                            st.success(f"✅ 卖出成功! 盈亏: ${result['realized_pnl']:+.2f} ← 子账户[{selected_account}]")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['error']}")
            else:
                st.info("暂无持仓可卖出")

        # 跨子账户调仓（A账户减仓 -> B账户加仓）
        with st.expander("↔ 跨子账户调仓", expanded=False):
            st.caption("将某子账户中的持仓部分转移到另一个子账户（先卖出，再买入）")
            source_account = st.selectbox("来源子账户", sub_names, key="rebalance_source_account")
            target_candidates = [x for x in sub_names if x != source_account] or [source_account]
            target_account = st.selectbox("目标子账户", target_candidates, key="rebalance_target_account")

            source_data = get_paper_account(source_account)
            source_positions = source_data.get("positions", []) if source_data else []
            if source_positions:
                pos_options = [f"{p['symbol']} | {p['market']} | {p['shares']}股" for p in source_positions]
                sel = st.selectbox("来源持仓", pos_options, key="rebalance_position_select")
                sel_symbol = sel.split(" | ")[0]
                sel_market = sel.split(" | ")[1]
                sel_pos = next((p for p in source_positions if p["symbol"] == sel_symbol and p["market"] == sel_market), None)
                max_move = int(sel_pos["shares"]) if sel_pos else 0
                move_shares = st.number_input("调仓股数", min_value=1, max_value=max_move if max_move > 0 else 1, value=max_move if max_move > 0 else 1, key="rebalance_move_shares")
                move_price = st.number_input("执行价格 (0=市价)", min_value=0.0, value=0.0, key="rebalance_move_price")

                if st.button("执行调仓", key="do_rebalance_accounts_btn"):
                    if source_account == target_account:
                        st.error("来源和目标子账户不能相同")
                    else:
                        px = move_price if move_price > 0 else None
                        sell_res = paper_sell(sel_symbol, int(move_shares), px, sel_market, source_account)
                        if not sell_res.get("success"):
                            st.error(f"❌ 来源账户卖出失败: {sell_res.get('error')}")
                        else:
                            buy_px = sell_res.get("price", px)
                            buy_res = paper_buy(sel_symbol, int(move_shares), buy_px, sel_market, target_account)
                            if buy_res.get("success"):
                                st.success(
                                    f"✅ 调仓完成: {sel_symbol} {move_shares}股 "
                                    f"[{source_account}] → [{target_account}]"
                                )
                                st.rerun()
                            else:
                                # 失败回滚：尝试买回来源账户
                                rollback = paper_buy(sel_symbol, int(move_shares), buy_px, sel_market, source_account)
                                if rollback.get("success"):
                                    st.error(f"❌ 目标账户买入失败，已回滚来源账户: {buy_res.get('error')}")
                                else:
                                    st.error(
                                        "❌ 目标账户买入失败且回滚失败，请立即手动检查。"
                                        f"目标错误: {buy_res.get('error')} | 回滚错误: {rollback.get('error')}"
                                    )
            else:
                st.info("来源子账户暂无持仓")

        # 单账户目标权重调仓（先预览再执行）
        with st.expander("🎯 目标权重调仓（单账户）", expanded=False):
            st.caption("为单个子账户设置目标权重，系统自动计算并执行买卖（先卖后买）")
            tw_account = st.selectbox("调仓账户", sub_names, index=sub_names.index(selected_account) if selected_account in sub_names else 0, key="tw_rebalance_account")
            tw_data = get_paper_account(tw_account)
            tw_positions = tw_data.get("positions", []) if tw_data else []

            if not tw_positions:
                st.info("该账户暂无持仓")
            else:
                total_equity = float(tw_data.get("total_equity", 0) or 0)
                investable_symbols = [p for p in tw_positions if (p.get("current_price") or 0) > 0]
                if not investable_symbols or total_equity <= 0:
                    st.warning("当前持仓缺少有效价格，暂无法计算目标权重")
                else:
                    mode = st.radio("调仓模式", ["一键等权", "手动权重"], horizontal=True, key="tw_rebalance_mode")
                    cash_reserve_pct = st.slider("现金保留 (%)", min_value=0, max_value=50, value=5, step=1, key="tw_cash_reserve_pct")
                    min_trade_value = st.number_input("最小交易金额 ($)", min_value=0.0, value=300.0, step=50.0, key="tw_min_trade_value")

                    target_weights = {}
                    remain_pct = max(0.0, 100.0 - float(cash_reserve_pct))
                    if mode == "一键等权":
                        each = remain_pct / len(investable_symbols)
                        for p in investable_symbols:
                            target_weights[p["symbol"]] = each
                    else:
                        st.caption("手动输入各持仓目标权重(%)，系统会自动按比例归一化到可投资比例")
                        raw_sum = 0.0
                        for p in investable_symbols:
                            curr_mv = float(p.get("market_value", 0) or 0)
                            curr_w = (curr_mv / total_equity * 100) if total_equity > 0 else 0.0
                            w = st.number_input(
                                f"{p['symbol']} 目标权重(%)",
                                min_value=0.0,
                                max_value=100.0,
                                value=float(round(curr_w, 2)),
                                step=0.5,
                                key=f"tw_weight_{tw_account}_{p['symbol']}_{p.get('market','US')}"
                            )
                            target_weights[p["symbol"]] = float(w)
                            raw_sum += float(w)

                        if raw_sum <= 0:
                            st.warning("目标权重全为0，将仅保留现金")
                            for k in list(target_weights.keys()):
                                target_weights[k] = 0.0
                        else:
                            scale = remain_pct / raw_sum
                            for k in list(target_weights.keys()):
                                target_weights[k] = target_weights[k] * scale

                    # 生成调仓计划
                    plan_rows = []
                    for p in investable_symbols:
                        sym = p["symbol"]
                        mkt = p.get("market", "US")
                        px = float(p.get("current_price", 0) or 0)
                        shares_now = int(p.get("shares", 0) or 0)
                        curr_value = float(p.get("market_value", shares_now * px) or 0)
                        tgt_value = total_equity * (target_weights.get(sym, 0.0) / 100.0)
                        delta_value = tgt_value - curr_value
                        action = "HOLD"
                        qty = 0
                        if abs(delta_value) >= float(min_trade_value) and px > 0:
                            if delta_value > 0:
                                qty = int(delta_value / px)
                                action = "BUY" if qty > 0 else "HOLD"
                            else:
                                qty = int(abs(delta_value) / px)
                                qty = min(qty, shares_now)
                                action = "SELL" if qty > 0 else "HOLD"

                        plan_rows.append({
                            "symbol": sym,
                            "market": mkt,
                            "current_shares": shares_now,
                            "current_weight_pct": (curr_value / total_equity * 100) if total_equity > 0 else 0.0,
                            "target_weight_pct": target_weights.get(sym, 0.0),
                            "delta_value": delta_value,
                            "action": action,
                            "shares": qty,
                            "ref_price": px,
                        })

                    plan_df = pd.DataFrame(plan_rows)
                    if not plan_df.empty:
                        view_df = plan_df.copy()
                        view_df["current_weight_pct"] = view_df["current_weight_pct"].map(lambda x: f"{x:.2f}%")
                        view_df["target_weight_pct"] = view_df["target_weight_pct"].map(lambda x: f"{x:.2f}%")
                        view_df["delta_value"] = view_df["delta_value"].map(lambda x: f"${x:,.2f}")
                        view_df["ref_price"] = view_df["ref_price"].map(lambda x: f"${x:,.2f}")
                        view_df = view_df.rename(columns={
                            "symbol": "代码", "market": "市场", "current_shares": "当前股数",
                            "current_weight_pct": "当前权重", "target_weight_pct": "目标权重",
                            "delta_value": "目标差额", "action": "动作", "shares": "执行股数", "ref_price": "参考价"
                        })
                        st.dataframe(view_df, hide_index=True, use_container_width=True)

                        actionable = plan_df[(plan_df["action"] != "HOLD") & (plan_df["shares"] > 0)]
                        st.caption(f"待执行指令: {len(actionable)} 条（先卖后买）")

                        if st.button("🚀 执行目标权重调仓", key="tw_execute_btn", disabled=len(actionable) == 0):
                            exec_errors = []
                            sell_orders = actionable[actionable["action"] == "SELL"]
                            buy_orders = actionable[actionable["action"] == "BUY"]

                            # 先卖后买，降低现金不足概率
                            for _, row in sell_orders.iterrows():
                                res = paper_sell(row["symbol"], int(row["shares"]), None, row["market"], tw_account)
                                if not res.get("success"):
                                    exec_errors.append(f"SELL {row['symbol']} 失败: {res.get('error')}")

                            for _, row in buy_orders.iterrows():
                                res = paper_buy(row["symbol"], int(row["shares"]), None, row["market"], tw_account)
                                if not res.get("success"):
                                    exec_errors.append(f"BUY {row['symbol']} 失败: {res.get('error')}")

                            if exec_errors:
                                st.error("部分指令执行失败:\n" + "\n".join(exec_errors[:8]))
                            else:
                                st.success(f"✅ 目标权重调仓完成（账户: {tw_account}）")
                            st.rerun()
        
        st.divider()
        
        # 模拟持仓列表
        st.subheader("📋 模拟持仓")
        
        if account['positions']:
            pos_data = []
            for p in account['positions']:
                pos_data.append({
                    '代码': p['symbol'],
                    '股数': p['shares'],
                    '成本': f"${p['avg_cost']:.2f}",
                    '现价': f"${p['current_price']:.2f}" if p.get('current_price') else '--',
                    '市值': f"${p['market_value']:,.2f}" if p.get('market_value') else '--',
                    '盈亏': f"${p['unrealized_pnl']:+,.2f}" if p.get('unrealized_pnl') else '--',
                    '盈亏%': f"{p['unrealized_pnl_pct']:+.2f}%" if p.get('unrealized_pnl_pct') else '--'
                })
            
            st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)
        else:
            st.info("暂无模拟持仓")

        # 全部子账户持仓总览（解决“买了但看不到”的困惑）
        with st.expander("🗂️ 全部子账户持仓总览", expanded=False):
            all_pos_rows = []
            for acc_name in sub_names:
                acc_data = get_paper_account(acc_name)
                if not acc_data:
                    continue
                for p in acc_data.get("positions", []):
                    all_pos_rows.append({
                        "子账户": acc_name,
                        "代码": p.get("symbol"),
                        "市场": p.get("market"),
                        "股数": int(p.get("shares", 0) or 0),
                        "成本": float(p.get("avg_cost", 0) or 0),
                        "现价": float(p.get("current_price", 0) or 0) if p.get("current_price") else None,
                        "市值": float(p.get("market_value", 0) or 0) if p.get("market_value") else 0.0,
                        "盈亏": float(p.get("unrealized_pnl", 0) or 0),
                        "盈亏%": float(p.get("unrealized_pnl_pct", 0) or 0),
                    })

            if all_pos_rows:
                all_df = pd.DataFrame(all_pos_rows)
                account_filter = st.selectbox(
                    "按子账户筛选",
                    ["全部"] + sub_names,
                    index=0,
                    key="all_paper_positions_account_filter"
                )
                if account_filter != "全部":
                    all_df = all_df[all_df["子账户"] == account_filter]

                # 美化显示
                show_df = all_df.copy()
                show_df["成本"] = show_df["成本"].map(lambda x: f"${x:.2f}")
                show_df["现价"] = show_df["现价"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "--")
                show_df["市值"] = show_df["市值"].map(lambda x: f"${x:,.2f}")
                show_df["盈亏"] = show_df["盈亏"].map(lambda x: f"${x:+,.2f}")
                show_df["盈亏%"] = show_df["盈亏%"].map(lambda x: f"{x:+.2f}%")

                st.dataframe(show_df, hide_index=True, use_container_width=True)
                st.caption(f"共 {len(show_df)} 条持仓记录")
            else:
                st.info("所有子账户均暂无持仓")
        
        # 交易记录
        with st.expander("📜 模拟交易记录", expanded=False):
            paper_trades = get_paper_trades(selected_account, limit=30)
            if paper_trades:
                trades_df = pd.DataFrame(paper_trades)
                display_cols = ['symbol', 'trade_type', 'price', 'shares', 'commission', 'trade_date', 'notes']
                available_cols = [c for c in display_cols if c in trades_df.columns]
                display_df = trades_df[available_cols].copy()
                display_df.columns = ['代码', '类型', '价格', '股数', '佣金', '日期', '备注'][:len(available_cols)]
                display_df['类型'] = display_df['类型'].map({'BUY': '🟢买入', 'SELL': '🔴卖出'})
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.info("暂无交易记录")
        
        # 权益曲线图
        st.subheader("📈 权益曲线")
        
        equity_curve = get_paper_equity_curve(selected_account)
        
        if not equity_curve.empty and len(equity_curve) > 1:
            import plotly.graph_objects as go
            
            fig_equity = go.Figure()
            
            # 总权益曲线
            fig_equity.add_trace(go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['total_equity'],
                mode='lines+markers',
                name='总权益',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            ))
            
            # 初始资金线
            initial = account['initial_capital']
            fig_equity.add_hline(y=initial, line_dash="dash", line_color="gray",
                                annotation_text=f"初始资金 ${initial:,.0f}")
            
            fig_equity.update_layout(
                title="账户权益变化",
                xaxis_title="日期",
                yaxis_title="权益 ($)",
                height=350,
                showlegend=True
            )
            
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # 收益率曲线
            col_ret, col_heat = st.columns(2)
            
            with col_ret:
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Bar(
                    x=equity_curve['date'],
                    y=equity_curve['return_pct'],
                    marker_color=['#4CAF50' if r >= 0 else '#F44336' for r in equity_curve['return_pct']],
                    name='累计收益率'
                ))
                fig_ret.update_layout(
                    title="累计收益率 (%)",
                    height=250,
                    showlegend=False
                )
                st.plotly_chart(fig_ret, use_container_width=True)
            
            with col_heat:
                # 月度收益热力图
                monthly = get_paper_monthly_returns(selected_account)
                if not monthly.empty:
                    import plotly.express as px
                    
                    # 创建透视表
                    pivot = monthly.pivot(index='year', columns='month', values='return_pct')
                    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
                    
                    fig_heat = px.imshow(
                        pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        aspect='auto',
                        title="月度收益热力图 (%)"
                    )
                    fig_heat.update_layout(height=250)
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("暂无足够数据生成热力图")
        else:
            st.info("开始交易后将显示权益曲线")
        
        # 已实现盈亏统计
        with st.expander("💰 已实现盈亏", expanded=False):
            realized = get_realized_pnl_history(selected_account)
            if realized:
                total_realized = sum(r['realized_pnl'] for r in realized)
                wins = len([r for r in realized if r['realized_pnl'] > 0])
                losses = len([r for r in realized if r['realized_pnl'] <= 0])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("已实现盈亏", f"${total_realized:+,.2f}")
                c2.metric("盈利笔数", f"{wins} 笔")
                c3.metric("亏损笔数", f"{losses} 笔")
                
                # 明细表
                realized_df = pd.DataFrame(realized)
                realized_df['realized_pnl'] = realized_df['realized_pnl'].apply(lambda x: f"${x:+,.2f}")
                realized_df.columns = ['日期', '代码', '价格', '股数', '盈亏']
                st.dataframe(realized_df, hide_index=True, use_container_width=True)
            else:
                st.info("暂无已实现盈亏")
        
        # 重置账户
        st.divider()
        if st.button("🔄 重置模拟账户", help="清空所有模拟持仓和交易记录，重置为初始资金"):
            reset_paper_account(selected_account)
            st.success("✅ 模拟账户已重置")
            st.rerun()


# --- 信号表现验证页面 (新增) ---

def render_signal_performance_page():
    """信号表现验证仪表盘 - 验证 BLUE 信号的历史有效性"""
    st.header("📊 信号表现验证 (Signal Performance)")
    st.info("支持快速回测与完整回测，并可按市值分层验证美股策略有效性")
    
    from services.backtest_service import (
        run_signal_backtest,
        run_full_signal_backtest,
        get_backtest_summary_table,
    )
    from datetime import datetime, timedelta
    
    # 侧边栏参数
    with st.sidebar:
        st.subheader("🎛️ 回测参数")
        
        # 市场选择
        market = st.radio("市场", ["US", "CN"], horizontal=True)
        
        # 日期范围
        col1, col2 = st.columns(2)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        with col1:
            start = st.date_input("开始日期", value=start_date)
        with col2:
            end = st.date_input("结束日期", value=end_date)
        
        # BLUE 阈值
        min_blue = st.slider("最低 BLUE 阈值", min_value=50, max_value=200, value=100, step=10)
        
        backtest_mode = st.radio("回测模式", ["快速回测", "完整回测"], horizontal=True)

        # 持仓周期（扩展选项）
        horizon_options = [3, 5, 7, 10, 15, 20, 30, 45, 60, 90]
        forward_days_list = st.multiselect(
            "持仓周期列表 (天)",
            options=horizon_options,
            default=[10, 20, 60],
            help="快速回测会对每个周期分别统计；完整回测使用其中最小值作为持有天数"
        )
        if not forward_days_list:
            forward_days_list = [10]
        primary_holding_days = min(forward_days_list)

        cap_filter_label = st.selectbox(
            "市值分层",
            ["全部", "Mega/Large", "Mid", "Small/Micro"],
            index=0,
            help="验证不同市值层策略表现差异"
        )
        cap_filter_map = {
            "全部": "all",
            "Mega/Large": "mega_large",
            "Mid": "mid",
            "Small/Micro": "small_micro",
        }
        cap_filter = cap_filter_map.get(cap_filter_label, "all")
        
        # 分析数量限制
        limit = st.number_input("最大分析数量", min_value=50, max_value=500, value=200, step=50)

        if backtest_mode == "完整回测":
            st.markdown("**完整回测参数**")
            initial_capital = st.number_input("初始资金", min_value=10000, max_value=1000000, value=100000, step=10000)
            max_positions = st.slider("最大持仓数", min_value=3, max_value=30, value=10)
            position_size_pct = st.slider("单仓位(%)", min_value=2, max_value=30, value=10) / 100.0
            commission = st.number_input("单边佣金", min_value=0.0, max_value=0.005, value=0.0005, format="%.4f")
            slippage = st.number_input("单边滑点", min_value=0.0, max_value=0.01, value=0.001, format="%.4f")
            run_walk_forward = st.checkbox("启用 Walk-forward", value=True)
        else:
            initial_capital = 100000
            max_positions = 10
            position_size_pct = 0.1
            commission = 0.0005
            slippage = 0.001
            run_walk_forward = False
        
        run_btn = st.button("🚀 开始验证", type="primary", use_container_width=True)
    
    # 使用说明
    if not run_btn:
        st.markdown("""
        ### 🎯 使用说明
        
        1. 在左侧设置 **回测参数**
        2. 点击 **开始验证** 按钮
        3. 查看 BLUE 信号的历史表现
        
        ---
        
        ### 📈 关键指标说明
        
        | 指标 | 说明 |
        |------|------|
        | **Win Rate** | 信号触发后盈利的比例 |
        | **Avg Return** | 平均每笔信号的收益率 |
        | **Sharpe Ratio** | 风险调整后收益 (>1 优秀) |
        | **Max Drawdown** | 最大回撤幅度 |
        | **Profit Factor** | 盈利/亏损比 (>1.5 优秀) |
        
        > ⚠️ **注意**: 需要信号日期后有足够的交易日数据才能计算收益
        """)
        return
    
    # 运行回测
    with st.spinner(f"正在分析 {market} 市场的 BLUE 信号..."):
        if backtest_mode == "完整回测":
            result = run_full_signal_backtest(
                start_date=start.strftime('%Y-%m-%d'),
                end_date=end.strftime('%Y-%m-%d'),
                market=market,
                min_blue=min_blue,
                holding_days=primary_holding_days,
                limit=limit,
                cap_filter=cap_filter,
                initial_capital=float(initial_capital),
                max_positions=int(max_positions),
                position_size_pct=float(position_size_pct),
                commission=float(commission),
                slippage=float(slippage),
                run_walk_forward=run_walk_forward
            )
        else:
            result = run_signal_backtest(
                start_date=start.strftime('%Y-%m-%d'),
                end_date=end.strftime('%Y-%m-%d'),
                market=market,
                min_blue=min_blue,
                forward_days=primary_holding_days,
                forward_days_list=forward_days_list,
                limit=limit,
                cap_filter=cap_filter
            )
    
    metrics = result.get('metrics', {})
    spy_metrics = result.get('spy_metrics', {})
    signals = result.get('signals', [])
    params = result.get('params', {})
    forward_days = int(params.get('forward_days', primary_holding_days))

    if backtest_mode == "完整回测":
        pf = result.get('portfolio_metrics', {}) or {}
        st.markdown("---")
        st.subheader("📊 完整回测概览")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("总收益率", f"{pf.get('total_return', 0):.2f}%")
        with c2:
            st.metric("胜率", f"{pf.get('win_rate', 0):.2f}%")
        with c3:
            st.metric("Sharpe", f"{pf.get('sharpe_ratio', 0):.2f}")
        with c4:
            st.metric("最大回撤", f"{pf.get('max_drawdown', 0):.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("交易数", pf.get('total_trades', 0))
        with c6:
            st.metric("组合持仓数", pf.get('num_positions', 0))
        with c7:
            st.metric("Sortino", f"{pf.get('sortino_ratio', 0):.2f}")
        with c8:
            st.metric("Calmar", f"{pf.get('calmar_ratio', 0):.2f}")

        walk = result.get('walk_forward', {})
        st.markdown("### 🧪 Walk-forward")
        if walk.get('error'):
            st.warning(f"Walk-forward 未完成: {walk.get('error')}")
        elif walk.get('status') == 'skipped':
            st.info(f"Walk-forward 跳过: {walk.get('reason')}")
        else:
            windows = walk.get('windows', [])
            if windows:
                wdf = pd.DataFrame(windows)
                st.dataframe(wdf, use_container_width=True, hide_index=True)

        trades = result.get('trades', [])
        if trades:
            st.markdown("### 📋 交易明细")
            tdf = pd.DataFrame(trades)
            st.dataframe(tdf, use_container_width=True, hide_index=True)

        with st.expander("🔧 回测参数"):
            st.json(params)
        return
    
    # 顶部摘要卡片
    st.markdown("---")
    st.subheader("📊 表现概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        win_rate = metrics.get('win_rate', 0)
        st.metric(
            "胜率 (Win Rate)",
            f"{win_rate:.1f}%",
            delta=f"vs SPY {spy_metrics.get('win_rate', 0):.1f}%" if spy_metrics else None,
            delta_color="normal" if win_rate > spy_metrics.get('win_rate', 0) else "inverse"
        )
    
    with col2:
        avg_ret = metrics.get('avg_return', 0)
        st.metric(
            "平均收益",
            f"{avg_ret:.2f}%",
            delta=f"vs SPY {spy_metrics.get('avg_return', 0):.2f}%" if spy_metrics else None,
            delta_color="normal" if avg_ret > spy_metrics.get('avg_return', 0) else "inverse"
        )
    
    with col3:
        sharpe = metrics.get('sharpe', 0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="优秀" if sharpe > 1 else ("良好" if sharpe > 0.5 else "待改进")
        )
    
    with col4:
        pf = metrics.get('profit_factor', 0)
        st.metric(
            "Profit Factor",
            f"{pf:.2f}",
            delta="优秀" if pf > 1.5 else ("一般" if pf > 1 else "亏损")
        )
    
    # 第二行指标
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("分析信号数", metrics.get('total_signals', 0))
    
    with col6:
        st.metric("盈利信号", metrics.get('winning_signals', 0))
    
    with col7:
        st.metric("亏损信号", metrics.get('losing_signals', 0))
    
    with col8:
        mdd = metrics.get('max_drawdown', 0)
        st.metric("最大回撤", f"{mdd:.2f}%", delta_color="inverse")
    
    # 对比表格
    st.markdown("---")
    st.subheader("📋 BLUE vs SPY 对比")
    
    summary_df = get_backtest_summary_table(result)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # 多周期表现（快速回测）
    metrics_by_h = result.get('metrics_by_horizon', {})
    if metrics_by_h:
        rows = []
        for h, m in metrics_by_h.items():
            rows.append({
                "周期": h,
                "胜率(%)": m.get("win_rate", 0),
                "平均收益(%)": m.get("avg_return", 0),
                "Sharpe": m.get("sharpe", 0),
                "最大回撤(%)": m.get("max_drawdown", 0),
                "样本": m.get("total_signals", 0),
            })
        if rows:
            st.markdown("### ⏱️ 多周期对比")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 市值分层表现
    cap_rows = result.get('cap_segment_metrics', [])
    if cap_rows:
        st.markdown("### 🧱 市值分层表现")
        cap_df = pd.DataFrame(cap_rows)
        cap_df = cap_df.rename(columns={
            "label": "市值层",
            "signals": "样本",
            "win_rate": "胜率(%)",
            "avg_return": "平均收益(%)",
            "sharpe": "Sharpe",
            "max_drawdown": "最大回撤(%)",
        })
        keep_cols = ["市值层", "样本", "胜率(%)", "平均收益(%)", "Sharpe", "最大回撤(%)"]
        cap_df = cap_df[[c for c in keep_cols if c in cap_df.columns]]
        st.dataframe(cap_df, use_container_width=True, hide_index=True)
    
    # 累积收益曲线图表
    st.markdown("---")
    st.subheader("📈 累积收益曲线 (Cumulative Returns)")
    
    from services.backtest_service import create_cumulative_returns_chart
    cumulative_chart = create_cumulative_returns_chart(result)
    st.plotly_chart(cumulative_chart, use_container_width=True)
    
    # 信号详情表
    if signals:
        st.markdown("---")
        st.subheader(f"📈 信号详情 ({len(signals)} 条)")
        
        # 转换为 DataFrame
        signals_df = pd.DataFrame(signals)
        
        # 根据 forward_days 动态确定列名
        ret_col = f'return_{forward_days}d'
        spy_ret_col = f'spy_return_{forward_days}d'
        alpha_col = f'alpha_{forward_days}d'
        
        # 格式化显示
        if ret_col in signals_df.columns:
            signals_df[f'{forward_days}d收益%'] = signals_df[ret_col].apply(
                lambda x: f"{x*100:.2f}%" if x is not None else "N/A"
            )
        if spy_ret_col in signals_df.columns:
            signals_df[f'SPY{forward_days}d%'] = signals_df[spy_ret_col].apply(
                lambda x: f"{x*100:.2f}%" if x is not None else "N/A"
            )
        if alpha_col in signals_df.columns:
            signals_df['Alpha%'] = signals_df[alpha_col].apply(
                lambda x: f"{x*100:.2f}%" if x is not None else "N/A"
            )
        
        display_cols = ['symbol', 'signal_date', 'blue_daily', 'price', 
                       f'{forward_days}d收益%', f'SPY{forward_days}d%', 'Alpha%']
        display_cols = [c for c in display_cols if c in signals_df.columns]
        
        # 重命名列
        rename_map = {
            'symbol': '代码',
            'signal_date': '信号日期',
            'blue_daily': 'Day BLUE',
            'price': '信号价格'
        }
        display_df = signals_df[display_cols].rename(columns=rename_map)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Day BLUE": st.column_config.ProgressColumn(
                    "Day BLUE",
                    format="%.0f",
                    min_value=0,
                    max_value=200
                )
            }
        )
    else:
        st.warning("未找到符合条件的信号。请调整日期范围或 BLUE 阈值。")
    
    # 参数摘要
    with st.expander("🔧 回测参数"):
        st.json(params)


# ==================== AI 决策仪表盘 ====================

def render_ai_dashboard_page():
    """AI 决策仪表盘页面 - Gemini 分析"""
    st.header("🤖 AI 决策仪表盘")
    st.caption("基于 Gemini 大模型的智能股票分析，生成一句话结论和检查清单")
    
    from ml.llm_intelligence import generate_stock_decision, check_llm_available
    
    # 检查 LLM 可用性
    llm_status = check_llm_available()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("股票代码", value="NVDA", key="ai_symbol").upper().strip()
        
    with col2:
        provider = st.selectbox("AI 模型", ["gemini", "openai"], index=0)
        if provider == "gemini" and not llm_status.get('gemini'):
            st.warning("Gemini 需要设置 GEMINI_API_KEY")
        elif provider == "openai" and not llm_status.get('openai'):
            st.warning("OpenAI 需要设置 OPENAI_API_KEY")
    
    if st.button("🔮 生成 AI 决策", key="gen_ai_decision"):
        with st.spinner(f"正在分析 {symbol}..."):
            try:
                # 获取股票数据
                from data_fetcher import get_us_stock_data
                from indicator_utils import calculate_blue_signal_series, MA
                
                df = get_us_stock_data(symbol, days=90)
                if df is None or len(df) < 30:
                    st.error("无法获取足够的股票数据")
                    return
                
                # 确保数据列存在
                df = df.reset_index(drop=True)
                
                # 计算均线
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA10'] = df['Close'].rolling(10).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                
                # 计算 BLUE 信号
                try:
                    blue_values = calculate_blue_signal_series(
                        df['Open'].values, df['High'].values, 
                        df['Low'].values, df['Close'].values
                    )
                    df['BLUE'] = blue_values
                except:
                    df['BLUE'] = 50  # 默认值
                
                latest = df.iloc[-1]
                price = float(latest['Close'])
                ma5 = float(latest['MA5']) if pd.notna(latest['MA5']) else price
                ma10 = float(latest['MA10']) if pd.notna(latest['MA10']) else price
                ma20 = float(latest['MA20']) if pd.notna(latest['MA20']) else price
                
                # 计算乖离率 (daily_stock_analysis 核心指标)
                bias_ma5 = (price - ma5) / ma5 * 100 if ma5 > 0 else 0
                
                # 判断均线排列
                ma_aligned = ma5 > ma10 > ma20  # 多头排列
                
                # 量比
                vol_ratio = float(latest['Volume']) / df['Volume'].rolling(5).mean().iloc[-1] if df['Volume'].rolling(5).mean().iloc[-1] > 0 else 1
                
                # 获取 BLUE 值
                blue_val = float(latest['BLUE']) if pd.notna(latest['BLUE']) and latest['BLUE'] != 0 else 50
                
                # 准备完整数据
                stock_data = {
                    'symbol': symbol,
                    'price': price,
                    'blue_daily': blue_val,
                    'blue_weekly': blue_val * 0.8,
                    'ma5': ma5,
                    'ma10': ma10,
                    'ma20': ma20,
                    'bias_ma5': bias_ma5,  # 乖离率
                    'ma_aligned': ma_aligned,  # 均线排列
                    'rsi': 50,
                    'volume_ratio': vol_ratio
                }
                
                # 显示技术数据预览
                with st.expander("📊 技术数据"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MA5", f"${ma5:.2f}")
                        st.metric("乖离率", f"{bias_ma5:+.2f}%", delta="危险" if bias_ma5 > 5 else None)
                    with col2:
                        st.metric("MA10", f"${ma10:.2f}")
                        st.metric("均线排列", "多头 ✅" if ma_aligned else "空头 ❌")
                    with col3:
                        st.metric("MA20", f"${ma20:.2f}")
                        st.metric("量比", f"{vol_ratio:.2f}x")
                
                # 生成决策
                from ml.llm_intelligence import LLMAnalyzer
                analyzer = LLMAnalyzer(provider=provider)
                result = analyzer.generate_decision_dashboard(stock_data)
                
                if 'error' in result:
                    st.error(f"分析失败: {result['error']}")
                    return
                
                # 显示结果
                st.success("✅ 分析完成")
                
                # 核心结论
                signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(result.get('signal', 'HOLD'), "🟡")
                st.markdown(f"### {signal_color} {result.get('verdict', '暂无结论')}")
                
                # 关键指标
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("信号", result.get('signal', 'N/A'))
                with col_b:
                    st.metric("置信度", f"{result.get('confidence', 0)}%")
                with col_c:
                    st.metric("入场价", f"${result.get('entry_price', 0):.2f}")
                with col_d:
                    st.metric("止损价", f"${result.get('stop_loss', 0):.2f}")
                
                # 目标价
                st.metric("🎯 目标价", f"${result.get('target_price', 0):.2f}")
                
                # 检查清单
                st.markdown("### ✅ 检查清单")
                checklist = result.get('checklist', [])
                for item in checklist:
                    status = item.get('status', '⚠️')
                    name = item.get('item', '')
                    detail = item.get('detail', '')
                    st.markdown(f"{status} **{name}**: {detail}")
                
                # 风险提示
                if result.get('risk_warning'):
                    st.warning(f"⚠️ {result.get('risk_warning')}")
                
            except Exception as e:
                st.error(f"分析出错: {str(e)}")


# ==================== 组合优化器 ====================

def render_portfolio_optimizer_page():
    """组合优化器页面 - Markowitz"""
    st.header("📐 组合优化器")
    st.caption("基于 Markowitz 均值-方差模型的资产配置优化")
    
    from research.portfolio_optimizer import optimize_portfolio_from_symbols
    
    # 输入股票
    symbols_input = st.text_input(
        "输入股票代码 (逗号分隔，最多10只)",
        value="AAPL, GOOGL, MSFT, NVDA, AMZN",
        key="portfolio_symbols"
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox("市场", ["US", "CN"], index=0, key="portfolio_market")
    with col2:
        days = st.number_input("历史天数", value=252, step=30, key="portfolio_days")
    
    if st.button("📊 优化组合", key="optimize_btn"):
        if len(symbols) < 2:
            st.error("至少需要2只股票")
            return
        
        with st.spinner("正在计算最优配置..."):
            try:
                result = optimize_portfolio_from_symbols(symbols, market=market, days=days)
                
                if 'error' in result:
                    st.error(f"优化失败: {result['error']}")
                    return
                
                st.success("✅ 优化完成")
                
                # 三种策略对比
                tab_sharpe, tab_vol, tab_parity = st.tabs(["📈 最大夏普", "🛡️ 最小波动", "⚖️ 风险平价"])
                
                with tab_sharpe:
                    sharpe = result.get('max_sharpe', {})
                    st.markdown("### 最大夏普比率组合")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("预期收益", f"{sharpe.get('expected_return', 0):.1f}%")
                    with col_b:
                        st.metric("波动率", f"{sharpe.get('volatility', 0):.1f}%")
                    with col_c:
                        st.metric("夏普比率", f"{sharpe.get('sharpe_ratio', 0):.2f}")
                    
                    st.markdown("**配置权重:**")
                    weights = sharpe.get('weights', {})
                    if weights:
                        import plotly.express as px
                        fig = px.pie(names=list(weights.keys()), values=list(weights.values()), 
                                     title="资产配置")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab_vol:
                    vol = result.get('min_vol', {})
                    st.markdown("### 最小波动率组合")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("预期收益", f"{vol.get('expected_return', 0):.1f}%")
                    with col_b:
                        st.metric("波动率", f"{vol.get('volatility', 0):.1f}%")
                    with col_c:
                        st.metric("夏普比率", f"{vol.get('sharpe_ratio', 0):.2f}")
                    
                    weights = vol.get('weights', {})
                    if weights:
                        st.dataframe(pd.DataFrame([weights]).T.rename(columns={0: '权重'}))
                
                with tab_parity:
                    parity = result.get('risk_parity', {})
                    st.markdown("### 风险平价组合")
                    st.caption("每个资产对总风险的贡献相等")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("预期收益", f"{parity.get('expected_return', 0):.1f}%")
                    with col_b:
                        st.metric("波动率", f"{parity.get('volatility', 0):.1f}%")
                    with col_c:
                        st.metric("夏普比率", f"{parity.get('sharpe_ratio', 0):.2f}")
                    
                    weights = parity.get('weights', {})
                    if weights:
                        st.dataframe(pd.DataFrame([weights]).T.rename(columns={0: '权重'}))
                
                # 相关性矩阵
                with st.expander("📊 相关性矩阵"):
                    corr = result.get('correlation', {})
                    if corr:
                        corr_df = pd.DataFrame(corr)
                        st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
                
            except Exception as e:
                st.error(f"优化出错: {str(e)}")


# ==================== 研究工具 ====================

def render_research_page():
    """研究工具页面 - 因子分析等"""
    st.header("🔬 研究工具")
    
    tab_factor, tab_ml, tab_charts = st.tabs(["📊 因子分析", "🤖 ML实验室", "📈 高级图表"])
    
    with tab_factor:
        st.subheader("📊 BLUE 因子 IC 分析")
        st.caption("分析 BLUE 信号对未来收益的预测能力")
        
        from research.factor_research import analyze_factors_from_scan
        
        col1, col2 = st.columns(2)
        with col1:
            market = st.selectbox("市场", ["US", "CN"], key="factor_market")
        
        if st.button("📈 分析 BLUE 因子", key="analyze_factor"):
            with st.spinner("正在分析..."):
                try:
                    result = analyze_factors_from_scan(market=market)
                    
                    if 'error' in result:
                        st.error(result['error'])
                        return
                    
                    st.success("✅ 分析完成")
                    
                    stats = result.get('stats', {})
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("平均 IC", f"{stats.get('mean_ic', 0):.4f}")
                    with col_b:
                        st.metric("IC_IR", f"{stats.get('ic_ir', 0):.4f}")
                    with col_c:
                        st.metric("IC 正向率", f"{stats.get('ic_positive_rate', 0):.1f}%")
                    with col_d:
                        st.metric("样本数", stats.get('n_periods', 0))
                    
                    # 解读
                    ic_ir = stats.get('ic_ir', 0)
                    if ic_ir > 0.5:
                        st.success("📈 BLUE 因子表现优秀 (IC_IR > 0.5)")
                    elif ic_ir > 0.3:
                        st.info("📊 BLUE 因子表现中等 (0.3 < IC_IR < 0.5)")
                    else:
                        st.warning("⚠️ BLUE 因子预测能力较弱 (IC_IR < 0.3)")
                    
                except Exception as e:
                    st.error(f"分析出错: {str(e)}")
    
    with tab_ml:
        # 保留原来的 ML 实验室内容
        render_ml_lab_page()
    
    with tab_charts:
        st.subheader("📈 高级图表工具")
        st.caption("专业级可视化分析工具")
        
        from advanced_charts import (
            create_multi_timeframe_heatmap,
            create_signal_radar_chart,
            create_drawdown_chart,
            create_volume_price_divergence_chart
        )
        from db.database import query_scan_results, get_scanned_dates
        
        chart_type = st.selectbox("选择图表类型", [
            "🔥 多周期共振热力图",
            "🎯 信号强度雷达图",
            "📉 回撤分析图",
            "📊 量价背离分析"
        ], key="chart_type_select")
        
        col1, col2 = st.columns(2)
        with col1:
            market = st.selectbox("市场", ["US", "CN"], key="adv_chart_market")
        
        if chart_type == "🔥 多周期共振热力图":
            if st.button("生成热力图", key="gen_heatmap"):
                with st.spinner("加载数据..."):
                    signals = query_scan_results(market=market, limit=30)
                    if signals:
                        data = {}
                        for s in signals:
                            symbol = s.get('symbol')
                            if symbol:
                                data[symbol] = {
                                    'day_blue': s.get('blue_daily', 0) or 0,
                                    'week_blue': s.get('blue_weekly', 0) or 0,
                                    'month_blue': s.get('blue_monthly', 0) or 0,
                                    'adx': s.get('adx', 0) or 0
                                }
                        fig = create_multi_timeframe_heatmap(data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无数据")
        
        elif chart_type == "🎯 信号强度雷达图":
            symbol = st.text_input("输入股票代码", value="AAPL", key="radar_symbol")
            if st.button("生成雷达图", key="gen_radar"):
                # 模拟获取信号数据
                signal_data = {
                    'blue_strength': np.random.randint(50, 100),
                    'trend_strength': np.random.randint(40, 90),
                    'volume_strength': np.random.randint(30, 80),
                    'chip_strength': np.random.randint(40, 85),
                    'momentum_strength': np.random.randint(45, 95)
                }
                fig = create_signal_radar_chart(signal_data)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("注: 数据为演示用途")
        
        elif chart_type == "📉 回撤分析图":
            if st.button("生成回撤图", key="gen_drawdown"):
                with st.spinner("计算..."):
                    from backtest.backtester import Backtester
                    signals = query_scan_results(market=market, limit=100)
                    if signals:
                        signals_df = pd.DataFrame(signals)
                        bt = Backtester()
                        result = bt.run_signal_backtest(signals_df, holding_days=10, market=market)
                        trades = result.get('trades', [])
                        if trades:
                            equity = [100000]
                            for t in trades:
                                equity.append(equity[-1] * (1 + t.get('pnl_pct', 0) / 100))
                            fig = create_drawdown_chart(equity)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无交易数据")
                    else:
                        st.warning("无信号数据")
        
        elif chart_type == "📊 量价背离分析":
            symbol = st.text_input("输入股票代码", value="AAPL", key="divergence_symbol")
            if st.button("分析量价背离", key="gen_divergence"):
                with st.spinner("加载数据..."):
                    from data_fetcher import get_us_stock_data, get_cn_stock_data
                    if market == "CN":
                        df = get_cn_stock_data(symbol, days=100)
                    else:
                        df = get_us_stock_data(symbol, days=100)
                    if df is not None and len(df) > 20:
                        fig = create_volume_price_divergence_chart(df, symbol)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无法获取数据")

# ==================== 回测实验室辅助函数 ====================

def render_parameter_lab():
    """参数实验室 - 批量回测验证不同参数组合"""
    import plotly.express as px
    from backtest.backtester import Backtester, backtest_blue_signals
    from db.database import query_scan_results, get_scanned_dates
    
    st.subheader("🔬 参数实验室")
    st.caption("基于历史扫描信号，批量验证不同参数组合的有效性")
    
    # --- 参数配置区 ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market = st.selectbox("选择市场", ["US", "CN"], index=0, key="param_lab_market")
        min_blue = st.slider("最低 BLUE 阈值", 50, 180, 100, step=10, key="param_lab_blue",
                            help="只测试 BLUE 值高于此阈值的信号")
    
    with col2:
        holding_days = st.slider("持有天数", 5, 30, 10, step=5, key="param_lab_days",
                                help="买入后固定持有的天数")
        signal_limit = st.slider("测试信号数量", 20, 200, 100, step=20, key="param_lab_limit",
                                help="最多测试多少个历史信号")
        backtest_mode = st.radio("回测模式", ["单信号回测", "组合回测"], horizontal=True, key="bt_mode",
                                help="组合回测模拟真实多仓操作")
    
    with col3:
        # 组合回测专用参数
        if backtest_mode == "组合回测":
            max_positions = st.slider("最大持仓数", 3, 15, 10, key="max_pos")
            position_pct = st.slider("单仓比例%", 5, 20, 10, key="pos_pct") / 100
        else:
            max_positions = 10
            position_pct = 0.1
        
        # 获取可用日期
        available_dates = _cached_scanned_dates(market=market)
        if available_dates:
            date_options = ["所有日期"] + available_dates[:30]
            selected_date = st.selectbox("指定日期 (可选)", date_options, key="param_lab_date")
        else:
            selected_date = "所有日期"
            st.warning("暂无扫描数据")
    
    # --- 运行回测 ---
    if st.button("🚀 开始批量回测", type="primary", key="run_param_lab"):
        with st.spinner("正在分析历史信号表现..."):
            try:
                # 获取历史信号
                scan_date = None if selected_date == "所有日期" else selected_date
                signals = query_scan_results(
                    scan_date=scan_date,
                    min_blue=min_blue,
                    market=market,
                    limit=signal_limit
                )
                
                if not signals:
                    st.warning("未找到符合条件的信号，请调整筛选条件")
                    return
                
                st.info(f"找到 **{len(signals)}** 个符合条件的信号，开始回测...")
                
                # 运行回测
                bt = Backtester()
                signals_df = pd.DataFrame(signals)
                
                if backtest_mode == "组合回测":
                    results = bt.run_portfolio_backtest(
                        signals_df, 
                        holding_days=holding_days, 
                        max_positions=max_positions,
                        position_size_pct=position_pct,
                        market=market
                    )
                else:
                    results = bt.run_signal_backtest(signals_df, holding_days=holding_days, market=market)
                
                # 获取基准对比
                benchmark = bt.compare_with_benchmark(
                    benchmark='SPY' if market == 'US' else '000001.SS',
                    period_days=30
                )
                
                # --- 显示结果 ---
                st.success("✅ 回测完成!")
                
                # 关键指标卡片
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("总交易数", results.get('total_trades', 0))
                m2.metric("胜率", f"{results.get('win_rate', 0):.1f}%", 
                         delta="好" if results.get('win_rate', 0) > 50 else "差")
                m3.metric("平均收益", f"{results.get('avg_return', 0):.2f}%")
                m4.metric("总收益", f"{results.get('total_return', 0):.2f}%")
                m5.metric("最大回撤", f"{results.get('max_drawdown', 0):.2f}%", delta_color="inverse")
                
                # 增强指标
                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                with col_a:
                    st.metric("夏普比率", f"{results.get('sharpe_ratio', 0):.2f}")
                with col_b:
                    st.metric("Sortino", f"{results.get('sortino_ratio', 0):.2f}",
                             help="只惩罚下行波动")
                with col_c:
                    st.metric("Calmar", f"{results.get('calmar_ratio', 0):.2f}",
                             help="年化收益/最大回撤")
                with col_d:
                    st.metric("信息比率", f"{results.get('information_ratio', 0):.2f}",
                             help="超额收益稳定性")
                with col_e:
                    alpha = benchmark.get('alpha', 0)
                    st.metric("Alpha", f"{alpha:+.2f}%",
                             delta="跑赢大盘" if alpha > 0 else "跑输大盘")
                
                # --- 资金曲线图 ---
                if results.get('trades'):
                    st.subheader("📈 模拟资金曲线")
                    
                    trades_df = pd.DataFrame(results['trades'])
                    trades_df['cumulative_return'] = (1 + trades_df['pnl_pct'] / 100).cumprod() * 100000
                    trades_df['trade_num'] = range(1, len(trades_df) + 1)
                    
                    fig = go.Figure()
                    
                    # 策略曲线
                    fig.add_trace(go.Scatter(
                        x=trades_df['trade_num'],
                        y=trades_df['cumulative_return'],
                        mode='lines+markers',
                        name='策略收益',
                        line=dict(color='#2196F3', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # 基准线 (初始资金)
                    fig.add_hline(y=100000, line_dash="dash", line_color="gray", 
                                 annotation_text="初始资金 $100,000")
                    
                    fig.update_layout(
                        title="累计收益曲线",
                        xaxis_title="交易序号",
                        yaxis_title="资金 ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- 收益分布图 ---
                    col_dist, col_monthly = st.columns(2)
                    
                    with col_dist:
                        st.subheader("📊 收益分布")
                        fig_dist = px.histogram(
                            trades_df, x='pnl_pct', nbins=20,
                            title="单笔收益分布",
                            labels={'pnl_pct': '收益率 (%)'},
                            color_discrete_sequence=['#4CAF50']
                        )
                        fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
                        fig_dist.update_layout(height=300)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col_monthly:
                        st.subheader("🗓️ 按月统计")
                        # 按月分组统计
                        trades_df['month'] = pd.to_datetime(trades_df['entry_date']).dt.to_period('M').astype(str)
                        monthly_stats = trades_df.groupby('month').agg({
                            'pnl_pct': ['mean', 'sum', 'count']
                        }).round(2)
                        monthly_stats.columns = ['平均收益%', '总收益%', '交易数']
                        monthly_stats = monthly_stats.reset_index()
                        monthly_stats.columns = ['月份', '平均收益%', '总收益%', '交易数']
                        
                        st.dataframe(monthly_stats, use_container_width=True, hide_index=True)
                    
                    # --- 交易明细 ---
                    with st.expander("📋 查看交易明细", expanded=False):
                        display_df = trades_df[['symbol', 'entry_date', 'entry_price', 
                                               'exit_price', 'holding_days', 'pnl_pct', 'win']].copy()
                        display_df.columns = ['股票', '入场日期', '入场价', '出场价', '持有天数', '收益%', '盈利']
                        display_df['盈利'] = display_df['盈利'].map({True: '✅', False: '❌'})
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"回测出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # --- 参数对比实验 ---
    st.divider()
    st.subheader("⚗️ 参数对比实验")
    st.caption("对比不同 BLUE 阈值的回测效果")
    
    if st.button("🧪 运行对比实验", key="run_compare"):
        with st.spinner("正在对比不同参数..."):
            thresholds = [60, 80, 100, 120, 150]
            comparison_results = []
            
            progress_bar = st.progress(0)
            
            for i, threshold in enumerate(thresholds):
                try:
                    result = backtest_blue_signals(
                        min_blue=threshold,
                        holding_days=10,
                        market=market,
                        limit=50
                    )
                    
                    if 'error' not in result:
                        comparison_results.append({
                            'BLUE阈值': threshold,
                            '交易数': result.get('total_trades', 0),
                            '胜率%': result.get('win_rate', 0),
                            '平均收益%': result.get('avg_return', 0),
                            '总收益%': result.get('total_return', 0),
                            '最大回撤%': result.get('max_drawdown', 0),
                            '夏普比率': result.get('sharpe_ratio', 0)
                        })
                except Exception as e:
                    st.warning(f"阈值 {threshold} 回测失败: {e}")
                
                progress_bar.progress((i + 1) / len(thresholds))
            
            if comparison_results:
                compare_df = pd.DataFrame(comparison_results)
                
                # 显示对比表格
                st.dataframe(
                    compare_df.style.background_gradient(subset=['胜率%', '平均收益%'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 可视化对比
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    x=compare_df['BLUE阈值'].astype(str),
                    y=compare_df['胜率%'],
                    name='胜率%',
                    marker_color='#4CAF50'
                ))
                fig_compare.add_trace(go.Scatter(
                    x=compare_df['BLUE阈值'].astype(str),
                    y=compare_df['平均收益%'],
                    mode='lines+markers',
                    name='平均收益%',
                    yaxis='y2',
                    line=dict(color='#2196F3', width=3)
                ))
                
                fig_compare.update_layout(
                    title="不同 BLUE 阈值的回测效果对比",
                    xaxis_title="BLUE 阈值",
                    yaxis=dict(title="胜率 (%)", side='left'),
                    yaxis2=dict(title="平均收益 (%)", side='right', overlaying='y'),
                    height=400
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # 最佳参数建议
                best_row = compare_df.loc[compare_df['平均收益%'].idxmax()]
                st.success(f"📌 **最佳参数建议**: BLUE 阈值 = **{int(best_row['BLUE阈值'])}**，"
                          f"平均收益 {best_row['平均收益%']:.2f}%，胜率 {best_row['胜率%']:.1f}%")
    
    # --- Walk-Forward 验证 ---
    st.divider()
    st.subheader("🔄 Walk-Forward 验证")
    st.caption("滚动训练/测试窗口，验证策略稳健性，防止过拟合")
    
    wf_col1, wf_col2 = st.columns(2)
    with wf_col1:
        train_days = st.slider("训练窗口 (天)", 30, 120, 60, step=15, key="wf_train")
    with wf_col2:
        test_days = st.slider("测试窗口 (天)", 10, 60, 20, step=10, key="wf_test")
    
    if st.button("🧪 运行 Walk-Forward 验证", key="run_wf"):
        with st.spinner("正在进行滚动验证..."):
            try:
                # 获取全部历史信号
                all_signals = query_scan_results(market=market, limit=500)
                
                if not all_signals or len(all_signals) < 50:
                    st.warning("历史数据不足，至少需要50条信号")
                else:
                    signals_df = pd.DataFrame(all_signals)
                    bt = Backtester()
                    wf_results = bt.walk_forward_backtest(
                        signals_df,
                        train_days=train_days,
                        test_days=test_days,
                        holding_days=10,
                        market=market
                    )
                    
                    if 'error' in wf_results:
                        st.warning(wf_results['error'])
                    else:
                        st.success(f"✅ 完成 **{wf_results['num_windows']}** 个滚动窗口验证!")
                        
                        # 汇总指标
                        wf_m1, wf_m2, wf_m3 = st.columns(3)
                        wf_m1.metric("平均胜率", f"{wf_results['avg_win_rate']:.1f}%")
                        wf_m2.metric("平均收益", f"{wf_results['avg_return']:.2f}%")
                        wf_m3.metric("平均夏普", f"{wf_results['avg_sharpe']:.2f}")
                        
                        # 窗口明细表
                        if wf_results.get('windows'):
                            windows_df = pd.DataFrame(wf_results['windows'])
                            display_cols = ['test_start', 'test_end', 'test_signals', 
                                          'test_win_rate', 'test_avg_return', 'test_sharpe']
                            windows_df = windows_df[display_cols]
                            windows_df.columns = ['测试开始', '测试结束', '信号数', '胜率%', '平均收益%', '夏普']
                            
                            st.dataframe(
                                windows_df.style.background_gradient(subset=['胜率%', '平均收益%'], cmap='RdYlGn'),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # 可视化各窗口表现
                            fig_wf = go.Figure()
                            fig_wf.add_trace(go.Bar(
                                x=[f"W{i+1}" for i in range(len(windows_df))],
                                y=windows_df['胜率%'],
                                name='胜率%',
                                marker_color='#4CAF50'
                            ))
                            fig_wf.add_trace(go.Scatter(
                                x=[f"W{i+1}" for i in range(len(windows_df))],
                                y=windows_df['平均收益%'],
                                mode='lines+markers',
                                name='平均收益%',
                                yaxis='y2',
                                line=dict(color='#2196F3', width=2)
                            ))
                            fig_wf.update_layout(
                                title="各窗口测试表现",
                                xaxis_title="窗口",
                                yaxis=dict(title="胜率%", side='left'),
                                yaxis2=dict(title="平均收益%", side='right', overlaying='y'),
                                height=350
                            )
                            st.plotly_chart(fig_wf, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Walk-Forward 验证出错: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # --- 蒙特卡洛模拟 ---
    st.divider()
    st.subheader("🎲 蒙特卡洛模拟")
    st.caption("通过随机抽样评估策略风险，计算盈利/破产概率")
    
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        num_sims = st.slider("模拟次数", 100, 2000, 500, step=100, key="mc_sims")
    with mc_col2:
        trades_per_sim = st.slider("每次模拟交易数", 20, 100, 50, step=10, key="mc_trades")
    with mc_col3:
        bankruptcy_pct = st.slider("破产阈值 (%)", 30, 70, 50, step=10, key="mc_bankrupt")
    
    if st.button("🎰 运行蒙特卡洛模拟", key="run_mc"):
        with st.spinner("正在进行蒙特卡洛模拟..."):
            try:
                from backtest.monte_carlo import monte_carlo_simulation, create_monte_carlo_charts
                
                # 获取历史交易数据
                all_signals = query_scan_results(market=market, limit=300)
                
                if not all_signals or len(all_signals) < 20:
                    st.warning("历史数据不足，至少需要20条信号")
                else:
                    # 先运行一次回测获取交易记录
                    signals_df = pd.DataFrame(all_signals)
                    bt = Backtester()
                    bt_result = bt.run_signal_backtest(signals_df, holding_days=10, market=market)
                    trades = bt_result.get('trades', [])
                    
                    if len(trades) < 10:
                        st.warning("有效交易数不足，无法进行模拟")
                    else:
                        # 运行蒙特卡洛
                        mc_result = monte_carlo_simulation(
                            trades,
                            num_simulations=num_sims,
                            trades_per_sim=trades_per_sim,
                            bankruptcy_threshold=bankruptcy_pct / 100
                        )
                        
                        if 'error' in mc_result:
                            st.warning(mc_result['error'])
                        else:
                            st.success(f"✅ 完成 **{num_sims}** 次模拟!")
                            
                            # 关键指标
                            mc_m1, mc_m2, mc_m3, mc_m4 = st.columns(4)
                            mc_m1.metric("盈利概率", f"{mc_result['profit_probability']:.1f}%",
                                        delta="好" if mc_result['profit_probability'] > 60 else "差")
                            mc_m2.metric("破产概率", f"{mc_result['bankruptcy_probability']:.1f}%",
                                        delta="低风险" if mc_result['bankruptcy_probability'] < 10 else "高风险",
                                        delta_color="inverse")
                            mc_m3.metric("平均收益", f"{mc_result['mean_return_pct']:.1f}%")
                            mc_m4.metric("平均最大回撤", f"{mc_result['mean_max_drawdown']:.1f}%")
                            
                            # 置信区间
                            st.markdown(f"""
                            **90% 置信区间**: 终值在 **${mc_result['ci_5']:,.0f}** ~ **${mc_result['ci_95']:,.0f}** 之间
                            
                            (初始资金 $100,000)
                            """)
                            
                            # 图表
                            charts = create_monte_carlo_charts(mc_result)
                            
                            if 'distribution' in charts:
                                st.plotly_chart(charts['distribution'], use_container_width=True)
                            
                            if 'curves' in charts:
                                st.plotly_chart(charts['curves'], use_container_width=True)
                            
                            if 'gauges' in charts:
                                st.plotly_chart(charts['gauges'], use_container_width=True)
                            
            except Exception as e:
                st.error(f"蒙特卡洛模拟出错: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_picks_performance_tab():
    """📈 机会表现 - 追踪历史选股表现"""
    st.subheader("📈 每日机会历史表现")
    st.caption("追踪每日扫描出的机会后续表现，分析哪些特征与成功相关")
    
    try:
        from strategies.picks_tracker import (
            PicksPerformanceTracker, FeatureAnalyzer,
            record_todays_picks
        )
        
        tracker = PicksPerformanceTracker()
        analyzer = FeatureAnalyzer(tracker)
        
        # 操作区
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 更新收益数据", help="为缺少收益的记录计算前向收益"):
                with st.spinner("正在更新..."):
                    result = tracker.batch_update_returns(limit=50)
                    st.success(f"✅ 更新完成: {result['updated']}/{result['total']}")
        
        with col2:
            days = st.selectbox("分析周期", [30, 60, 90, 180], index=1)
        
        with col3:
            market = st.selectbox("市场", ["US", "CN", "全部"], index=0)
        
        # 表现汇总
        st.markdown("### 📊 表现汇总")
        
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        summary = tracker.get_performance_summary(
            start_date=start_date,
            end_date=end_date,
            market=market if market != "全部" else None
        )
        
        if summary.get('total_picks', 0) > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总机会数", summary.get('total_picks', 0))
            m2.metric("平均5日收益", f"{summary.get('avg_return_d5', 0)}%")
            m3.metric("5日胜率", f"{summary.get('win_rate_d5', 0)}%")
            m4.metric("平均最大涨幅", f"{summary.get('avg_max_gain', 'N/A')}%")
            
            # 最佳/最差选股
            col_best, col_worst = st.columns(2)
            with col_best:
                best = summary.get('best_pick')
                if best:
                    st.success(f"🏆 最佳: {best.get('symbol')} ({best.get('pick_date')}) +{best.get('return_d5')}%")
            with col_worst:
                worst = summary.get('worst_pick')
                if worst:
                    st.error(f"😢 最差: {worst.get('symbol')} ({worst.get('pick_date')}) {worst.get('return_d5')}%")
        else:
            st.info("📭 暂无足够的历史数据，请先记录每日机会")
        
        st.divider()
        
        # 特征分析
        st.markdown("### 🔬 特征重要性分析")
        
        importance = analyzer.feature_importance()
        
        if importance.get('n_samples', 0) > 20:
            # 相关性表
            corr = importance.get('correlations', {})
            if corr:
                corr_df = pd.DataFrame([
                    {'特征': k, '与5日收益相关性': v, 
                     '解读': '✅ 正相关' if v > 0.1 else ('❌ 负相关' if v < -0.1 else '➖ 弱相关')}
                    for k, v in corr.items()
                ])
                corr_df = corr_df.sort_values('与5日收益相关性', ascending=False)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            # 分类特征分析
            cat_analysis = importance.get('categorical_analysis', {})
            if cat_analysis:
                st.markdown("**分类特征影响:**")
                if 'heima_effect' in cat_analysis:
                    he = cat_analysis['heima_effect']
                    st.write(f"🐴 黑马信号: 有黑马 {he.get('heima_avg')}% vs 无黑马 {he.get('non_heima_avg')}% (提升 {he.get('lift')}%)")
                
                if 'new_discovery_effect' in cat_analysis:
                    ne = cat_analysis['new_discovery_effect']
                    st.write(f"🆕 新发现: 新 {ne.get('new_avg')}% vs 老 {ne.get('old_avg')}% (提升 {ne.get('lift')}%)")
        else:
            st.warning(f"样本不足 ({importance.get('n_samples', 0)} < 20)，无法进行特征分析")
        
        st.divider()
        
        # 策略有效性
        st.markdown("### 🎯 策略有效性排名")
        
        strategies = analyzer.strategy_effectiveness()
        
        if strategies:
            strategy_df = pd.DataFrame([
                {
                    '策略': name,
                    '选股数': stats['total_picks'],
                    '平均收益': f"{stats['avg_return_d5']}%",
                    '胜率': f"{stats['win_rate']}%",
                    'Sharpe-like': stats['sharpe_like'],
                    '最佳': f"{stats['best']}%",
                    '最差': f"{stats['worst']}%"
                }
                for name, stats in strategies.items()
            ])
            st.dataframe(strategy_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无策略表现数据")
            
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_strategy_optimizer_tab():
    """🎯 策略优化 - 自动寻找最优参数"""
    st.subheader("🎯 策略参数优化器")
    st.caption("通过历史数据自动寻找最优策略参数组合")
    
    try:
        from strategies.optimizer import (
            StrategyOptimizer, ContinuousOptimizer,
            StrategyConfig, optimize_strategies
        )
        
        optimizer = StrategyOptimizer()
        
        # 当前最优配置
        st.markdown("### 🏆 当前最优配置")
        
        best_config = optimizer.get_best_config()
        if best_config:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("策略名称", best_config.name[:15])
            col2.metric("BLUE日线阈值", best_config.blue_daily_min)
            col3.metric("BLUE周线阈值", best_config.blue_weekly_min)
            col4.metric("ADX阈值", best_config.adx_min)
            
            with st.expander("📋 完整配置"):
                st.json(best_config.to_dict())
        else:
            st.info("暂无保存的最优配置，请运行优化")
        
        st.divider()
        
        # 优化选项
        st.markdown("### 🔬 运行优化")
        
        opt_type = st.radio("优化方式", [
            "📊 比较预定义策略", 
            "🔍 网格搜索 (耗时较长)"
        ], horizontal=True)
        
        if st.button("🚀 开始优化", type="primary"):
            with st.spinner("正在优化策略参数..."):
                if "预定义" in opt_type:
                    results = optimizer.run_template_comparison()
                else:
                    results = optimizer.run_grid_search()
                
                if results:
                    st.success(f"✅ 优化完成！测试了 {len(results)} 种配置")
                    
                    # 显示结果表
                    results_df = pd.DataFrame([
                        {
                            '排名': r.rank,
                            '策略': r.config.name[:25],
                            '样本数': r.metrics.get('n_samples', 0),
                            '平均收益': f"{r.metrics.get('avg_return', 0)}%",
                            '胜率': f"{r.metrics.get('win_rate', 0)}%",
                            'Sharpe': r.metrics.get('sharpe_like', 0),
                            '综合得分': round(r.score, 1)
                        }
                        for r in results[:20]
                    ])
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # 保存最优
                    if st.button("💾 保存最优配置"):
                        if optimizer.save_best_config(results[0]):
                            st.success("✅ 已保存最优配置")
                        else:
                            st.error("保存失败")
                else:
                    st.warning("优化未产生有效结果，可能数据不足")
        
        st.divider()
        
        # 预定义策略模板
        st.markdown("### 📚 预定义策略模板")
        
        templates = StrategyOptimizer.STRATEGY_TEMPLATES
        template_df = pd.DataFrame([
            {
                '策略名称': name,
                'BLUE日线': cfg.blue_daily_min,
                'BLUE周线': cfg.blue_weekly_min,
                'ADX': cfg.adx_min,
                '黑马': '✅' if cfg.require_heima else '',
                '掘地': '✅' if cfg.require_juedi else '',
                '止损': f"{cfg.stop_loss_pct}%",
                '止盈': f"{cfg.take_profit_pct}%"
            }
            for name, cfg in templates.items()
        ])
        
        st.dataframe(template_df, use_container_width=True, hide_index=True)

        # 悬停查看策略细节（比表格更直观）
        strategy_desc_map = {
            "BLUE_强势": "强趋势跟随：更高日/周BLUE + 中高ADX，偏进攻。",
            "BLUE_保守": "三周期共振：日周月BLUE+更紧止损，偏稳健。",
            "黑马猎手": "优先黑马信号：捕捉加速段，容忍更高波动。",
            "掘地反攻": "优先掘地反转：左侧反攻，需更严格风控。",
            "三重共振": "日周月同向确认：牺牲频率换稳定性。",
            "新股狩猎": "新发现+高成交额：偏事件驱动，周转更快。",
            "筹码精选": "结合筹码结构：重视突破/密集区，过滤假信号。",
            "高ADX趋势": "只做高趋势强度：减少震荡区交易。",
        }
        hover_df = template_df.copy()
        hover_df["说明"] = hover_df["策略名称"].map(lambda x: strategy_desc_map.get(x, "自定义策略模板"))
        hover_df["黑马"] = hover_df["黑马"].replace({"": "否", "✅": "是"})
        hover_df["掘地"] = hover_df["掘地"].replace({"": "否", "✅": "是"})
        hover_df["止损值"] = hover_df["止损"].str.replace("%", "", regex=False).astype(float)

        st.markdown("#### 🖱️ 策略悬停详情图")
        st.caption("鼠标悬停查看每个策略的具体逻辑与参数（点越大表示止损越宽）")
        fig_hover = px.scatter(
            hover_df,
            x="BLUE日线",
            y="ADX",
            color="策略名称",
            size="止损值",
            hover_name="策略名称",
            hover_data={
                "说明": True,
                "BLUE周线": True,
                "黑马": True,
                "掘地": True,
                "止损": True,
                "止盈": True,
                "策略名称": False,
                "BLUE日线": True,
                "ADX": True,
                "止损值": False,
            },
            height=430,
        )
        fig_hover.update_layout(
            xaxis_title="BLUE日线阈值",
            yaxis_title="ADX阈值",
            legend_title="策略",
            hovermode="closest",
        )
        fig_hover.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "BLUE日线: %{x}<br>"
                "ADX: %{y}<br>"
                "说明: %{customdata[0]}<br>"
                "BLUE周线: %{customdata[1]}<br>"
                "黑马: %{customdata[2]} | 掘地: %{customdata[3]}<br>"
                "止损: %{customdata[4]} | 止盈: %{customdata[5]}<extra></extra>"
            )
        )
        st.plotly_chart(fig_hover, use_container_width=True)

        # 云端/移动端有时悬停不稳定，提供同等信息的手动查看兜底
        st.caption("若悬停无反应，可用下方选择器查看同样的策略细节")
        detail_pick = st.selectbox(
            "策略详情（手动查看）",
            options=hover_df["策略名称"].tolist(),
            key="strategy_hover_fallback_pick",
        )
        detail_row = hover_df[hover_df["策略名称"] == detail_pick].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("BLUE日线", int(detail_row["BLUE日线"]))
        d2.metric("BLUE周线", int(detail_row["BLUE周线"]))
        d3.metric("ADX", int(detail_row["ADX"]))
        d4.metric("止损/止盈", f"{detail_row['止损']} / {detail_row['止盈']}")
        st.info(
            f"{detail_pick}: {detail_row['说明']} | 黑马: {detail_row['黑马']} | 掘地: {detail_row['掘地']}"
        )

        st.divider()
        st.markdown("### 🧭 各策略最优参数与极致点")
        st.caption("逐个策略做参数网格评估，给出最高胜率参数、最高收益参数，以及“放松/收紧”后的衰减情况")

        c1, c2, c3 = st.columns(3)
        with c1:
            lookback_days = st.selectbox("评估窗口", [60, 120, 180, 360, 9999], index=2, key="opt_extreme_days")
        with c2:
            min_samples = st.slider("最小样本数", 10, 300, 40, step=10, key="opt_extreme_min_samples")
        with c3:
            market_scope = st.selectbox("市场过滤", ["全部", "US", "CN"], index=0, key="opt_extreme_market")

        if st.button("🧪 生成策略极致报告", key="run_extreme_report"):
            with st.spinner("正在评估各策略参数组合..."):
                df_hist = optimizer._load_historical_data()
                history_source = "optimizer_table"

                # 兜底：若优化器历史表为空，则回退到 candidate_tracking 构造评估样本
                if df_hist is None or df_hist.empty:
                    try:
                        from services.candidate_tracking_service import get_candidate_tracking_rows
                        tr_rows = get_candidate_tracking_rows(
                            market=None if market_scope == "全部" else market_scope,
                            days_back=max(365, int(lookback_days) if int(lookback_days) < 9999 else 720),
                        ) or []
                        if tr_rows:
                            mapped = []
                            for r in tr_rows:
                                ret_d5 = r.get("pnl_d5")
                                if ret_d5 is None:
                                    ret_d5 = r.get("pnl_pct")
                                mapped.append({
                                    "symbol": r.get("symbol"),
                                    "pick_date": r.get("signal_date"),
                                    "market": r.get("market"),
                                    "blue_daily": r.get("blue_daily"),
                                    "blue_weekly": r.get("blue_weekly"),
                                    "blue_monthly": r.get("blue_monthly"),
                                    "adx": r.get("adx"),
                                    "turnover": r.get("turnover_m"),
                                    "is_heima": bool(r.get("heima_daily") or r.get("heima_weekly") or r.get("heima_monthly")),
                                    "is_juedi": bool(r.get("juedi_daily") or r.get("juedi_weekly") or r.get("juedi_monthly")),
                                    "return_d5": ret_d5,
                                })
                            df_hist = pd.DataFrame(mapped)
                            history_source = "candidate_tracking"
                    except Exception:
                        pass

                if df_hist is None or df_hist.empty:
                    st.warning("历史样本为空，无法生成策略极致报告。请先在“每日工作台/组合追踪”累计并刷新候选追踪数据。")
                else:
                    st.caption(f"样本来源: {history_source} | 原始样本: {len(df_hist)}")
                    work_df = df_hist.copy()

                    # 按市场筛选
                    if market_scope != "全部" and "market" in work_df.columns:
                        work_df = work_df[work_df["market"].astype(str) == market_scope]

                    # 按时间窗口筛选
                    if lookback_days < 9999 and "pick_date" in work_df.columns:
                        cutoff = (datetime.now() - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
                        try:
                            work_df = work_df[work_df["pick_date"].astype(str) >= cutoff]
                        except Exception:
                            pass

                    if work_df.empty:
                        st.warning("筛选后样本为空，请放宽市场或时间窗口")
                    else:
                        eval_rows = []
                        for strat_name, base_cfg in templates.items():
                            day_vals = sorted(set([
                                max(40, int(base_cfg.blue_daily_min - 40)),
                                max(40, int(base_cfg.blue_daily_min - 20)),
                                int(base_cfg.blue_daily_min),
                                min(260, int(base_cfg.blue_daily_min + 20)),
                                min(260, int(base_cfg.blue_daily_min + 40)),
                            ]))
                            week_vals = sorted(set([
                                max(0, int(base_cfg.blue_weekly_min - 40)),
                                max(0, int(base_cfg.blue_weekly_min - 20)),
                                int(base_cfg.blue_weekly_min),
                                min(180, int(base_cfg.blue_weekly_min + 20)),
                                min(180, int(base_cfg.blue_weekly_min + 40)),
                            ]))
                            adx_vals = sorted(set([
                                max(10, int(base_cfg.adx_min - 10)),
                                max(10, int(base_cfg.adx_min - 5)),
                                int(base_cfg.adx_min),
                                min(80, int(base_cfg.adx_min + 5)),
                                min(80, int(base_cfg.adx_min + 10)),
                            ]))

                            for d_val in day_vals:
                                for w_val in week_vals:
                                    for a_val in adx_vals:
                                        cfg = StrategyConfig(
                                            name=f"{strat_name}_d{d_val}_w{w_val}_a{a_val}",
                                            blue_daily_min=float(d_val),
                                            blue_daily_max=float(base_cfg.blue_daily_max),
                                            blue_weekly_min=float(w_val),
                                            blue_monthly_min=float(base_cfg.blue_monthly_min),
                                            adx_min=float(a_val),
                                            adx_max=float(base_cfg.adx_max),
                                            turnover_min=float(base_cfg.turnover_min),
                                            require_heima=bool(base_cfg.require_heima),
                                            require_juedi=bool(base_cfg.require_juedi),
                                            require_new_discovery=bool(base_cfg.require_new_discovery),
                                            chip_patterns=list(base_cfg.chip_patterns or []),
                                            max_positions=int(base_cfg.max_positions),
                                            position_size_pct=float(base_cfg.position_size_pct),
                                            stop_loss_pct=float(base_cfg.stop_loss_pct),
                                            take_profit_pct=float(base_cfg.take_profit_pct),
                                        )
                                        result = optimizer.evaluate_strategy(cfg, work_df)
                                        metrics = result.metrics or {}
                                        n_samples = int(metrics.get("n_samples", 0) or 0)
                                        if n_samples < int(min_samples):
                                            continue
                                        eval_rows.append({
                                            "策略": strat_name,
                                            "BLUE日线": int(d_val),
                                            "BLUE周线": int(w_val),
                                            "ADX": int(a_val),
                                            "样本数": n_samples,
                                            "胜率(%)": float(metrics.get("win_rate", 0) or 0),
                                            "平均收益(%)": float(metrics.get("avg_return", 0) or 0),
                                            "Sharpe": float(metrics.get("sharpe_like", 0) or 0),
                                            "综合得分": float(result.score or 0),
                                        })

                        if not eval_rows:
                            st.warning("没有满足最小样本数的参数组合，请降低最小样本或扩大窗口")
                        else:
                            eval_df = pd.DataFrame(eval_rows)
                            st.caption(f"已评估组合数: {len(eval_df)}")

                            summary_rows = []
                            for strat_name in eval_df["策略"].unique():
                                sub = eval_df[eval_df["策略"] == strat_name].copy()
                                if sub.empty:
                                    continue

                                best_win = sub.sort_values(["胜率(%)", "平均收益(%)", "综合得分"], ascending=False).iloc[0]
                                best_ret = sub.sort_values(["平均收益(%)", "胜率(%)", "综合得分"], ascending=False).iloc[0]
                                best_score = sub.sort_values(["综合得分", "胜率(%)", "平均收益(%)"], ascending=False).iloc[0]

                                lower = sub[sub["BLUE日线"] < best_score["BLUE日线"]]
                                upper = sub[sub["BLUE日线"] > best_score["BLUE日线"]]

                                left_neighbor = None
                                right_neighbor = None
                                if not lower.empty:
                                    left_neighbor = lower.assign(
                                        _dist=(best_score["BLUE日线"] - lower["BLUE日线"]).abs()
                                    ).sort_values(["_dist", "综合得分"], ascending=[True, False]).iloc[0]
                                if not upper.empty:
                                    right_neighbor = upper.assign(
                                        _dist=(upper["BLUE日线"] - best_score["BLUE日线"]).abs()
                                    ).sort_values(["_dist", "综合得分"], ascending=[True, False]).iloc[0]

                                def _delta_text(nei):
                                    if nei is None:
                                        return "无样本"
                                    d_win = float(nei["胜率(%)"]) - float(best_score["胜率(%)"])
                                    d_ret = float(nei["平均收益(%)"]) - float(best_score["平均收益(%)"])
                                    d_n = int(nei["样本数"]) - int(best_score["样本数"])
                                    return f"胜率{d_win:+.1f} 收益{d_ret:+.2f} 样本{d_n:+d}"

                                summary_rows.append({
                                    "策略": strat_name,
                                    "最高胜率参数": f"D{int(best_win['BLUE日线'])}/W{int(best_win['BLUE周线'])}/ADX{int(best_win['ADX'])}",
                                    "最高胜率(%)": round(float(best_win["胜率(%)"]), 1),
                                    "最高收益参数": f"D{int(best_ret['BLUE日线'])}/W{int(best_ret['BLUE周线'])}/ADX{int(best_ret['ADX'])}",
                                    "最高收益(%)": round(float(best_ret["平均收益(%)"]), 2),
                                    "极致点参数(综合)": f"D{int(best_score['BLUE日线'])}/W{int(best_score['BLUE周线'])}/ADX{int(best_score['ADX'])}",
                                    "极致点样本数": int(best_score["样本数"]),
                                    "放松后变化": _delta_text(left_neighbor),
                                    "收紧后变化": _delta_text(right_neighbor),
                                })

                            summary_df = pd.DataFrame(summary_rows).sort_values("最高胜率(%)", ascending=False)
                            st.markdown("#### 1) 每个策略的最优参数与极致点")
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)

                            st.markdown("#### 2) 全部策略参数Top榜（按胜率）")
                            top_df = eval_df.sort_values(["胜率(%)", "平均收益(%)", "样本数"], ascending=False).head(30).copy()
                            top_df["参数组合"] = top_df.apply(
                                lambda r: f"D{int(r['BLUE日线'])}/W{int(r['BLUE周线'])}/ADX{int(r['ADX'])}", axis=1
                            )
                            st.dataframe(
                                top_df[["策略", "参数组合", "样本数", "胜率(%)", "平均收益(%)", "Sharpe", "综合得分"]],
                                use_container_width=True,
                                hide_index=True
                            )

                            st.markdown("#### 3) 策略参数路径（宽松 → 极致 → 收紧）")
                            strat_list = sorted(eval_df["策略"].unique().tolist())
                            selected_path_strat = st.selectbox(
                                "选择要查看路径的策略",
                                strat_list,
                                index=0,
                                key="opt_extreme_path_strategy",
                            )
                            sub_path = eval_df[eval_df["策略"] == selected_path_strat].copy()
                            if sub_path.empty:
                                st.info("该策略暂无参数路径数据")
                            else:
                                peak_row = sub_path.sort_values(
                                    ["综合得分", "胜率(%)", "平均收益(%)"],
                                    ascending=False
                                ).iloc[0]
                                peak_blue = int(peak_row["BLUE日线"])
                                sub_path["阶段"] = sub_path["BLUE日线"].apply(
                                    lambda v: "宽松段" if int(v) < peak_blue else ("收紧段" if int(v) > peak_blue else "极致点")
                                )
                                path_cols = ["阶段", "BLUE日线", "BLUE周线", "ADX", "样本数", "胜率(%)", "平均收益(%)", "Sharpe", "综合得分"]
                                path_df = sub_path[path_cols].sort_values(
                                    ["BLUE日线", "BLUE周线", "ADX"],
                                    ascending=True
                                )
                                st.dataframe(path_df, use_container_width=True, hide_index=True)

                                fig_path = px.line(
                                    path_df,
                                    x="BLUE日线",
                                    y="胜率(%)",
                                    color="阶段",
                                    markers=True,
                                    hover_data=["BLUE周线", "ADX", "样本数", "平均收益(%)", "Sharpe", "综合得分"],
                                    title=f"{selected_path_strat} 参数路径 - 胜率随 BLUE日线 变化",
                                )
                                st.plotly_chart(fig_path, use_container_width=True)
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_historical_review():
    """历史复盘 - 查看某天信号的后续表现"""
    from services.signal_tracker_service import get_signal_performance_summary
    from db.database import get_scanned_dates
    
    st.subheader("📊 历史复盘")
    st.caption("选择一个历史扫描日期，查看当天信号的后续表现")
    
    col1, col2 = st.columns(2)
    
    with col1:
        market = st.selectbox("市场", ["US", "CN"], index=0, key="review_market")
    
    with col2:
        dates = _cached_scanned_dates(market=market)
        if not dates:
            st.warning("暂无扫描数据")
            return
        selected_date = st.selectbox("选择扫描日期", dates[:30], key="review_date")
    
    if st.button("📈 分析信号表现", type="primary", key="run_review"):
        with st.spinner(f"正在分析 {selected_date} 的信号表现..."):
            try:
                summary = get_signal_performance_summary(selected_date, market)
                
                if not summary:
                    st.warning("未找到该日期的信号数据")
                    return
                
                # 显示统计摘要
                st.success(f"✅ 分析完成！共 {summary.get('total_signals', 0)} 个信号")
                
                # 关键指标
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("5日胜率", f"{summary.get('win_rate_5d', 0):.1f}%")
                m2.metric("10日胜率", f"{summary.get('win_rate_10d', 0):.1f}%")
                m3.metric("20日胜率", f"{summary.get('win_rate_20d', 0):.1f}%")
                m4.metric("大赚 (>10%)", f"{summary.get('big_win_20d', 0)} 只")
                
                # 平均收益
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("5日平均", f"{summary.get('avg_5d', 0):+.2f}%")
                col_b.metric("10日平均", f"{summary.get('avg_10d', 0):+.2f}%")
                col_c.metric("20日平均", f"{summary.get('avg_20d', 0):+.2f}%")
                
                # 详细表格
                if summary.get('details'):
                    st.subheader("📋 信号明细")
                    
                    details_df = pd.DataFrame(summary['details'])
                    
                    # 选择显示的列
                    display_cols = ['symbol', 'name', 'entry_price', 'return_5d', 
                                   'return_10d', 'return_20d', 'max_gain', 'max_drawdown']
                    available_cols = [c for c in display_cols if c in details_df.columns]
                    
                    if available_cols:
                        display_df = details_df[available_cols].copy()
                        display_df.columns = ['股票', '名称', '入场价', '5日收益%', 
                                             '10日收益%', '20日收益%', '最大涨幅%', '最大回撤%'][:len(available_cols)]
                        
                        # 颜色编码
                        def color_returns(val):
                            if pd.isna(val):
                                return ''
                            try:
                                v = float(val)
                                if v > 0:
                                    return 'color: green'
                                elif v < 0:
                                    return 'color: red'
                            except:
                                pass
                            return ''
                        
                        st.dataframe(
                            display_df.style.applymap(color_returns, 
                                                     subset=[c for c in display_df.columns if '收益' in c or '涨幅' in c or '回撤' in c]),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # 收益分布图
                        if 'return_20d' in details_df.columns:
                            import plotly.express as px
                            fig = px.histogram(
                                details_df.dropna(subset=['return_20d']),
                                x='return_20d',
                                nbins=15,
                                title=f"{selected_date} 信号的 20 日收益分布",
                                labels={'return_20d': '20日收益率 (%)'}
                            )
                            fig.add_vline(x=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"分析出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_backtest_page():
    st.header("🧪 策略回测实验室 (Strategy Lab)")
    
    tab_param_lab, tab_single, tab_risk, tab_review, tab_picks, tab_optimizer = st.tabs([
        "🔬 参数实验室", 
        "📈 单股回测", 
        "🛡️ 风控计算器",
        "📊 历史复盘",
        "📈 机会表现",
        "🎯 策略优化"
    ])
    
    # === 参数实验室 Tab (新增) ===
    with tab_param_lab:
        render_parameter_lab()
    
    # === 机会表现 Tab (新增) ===
    with tab_picks:
        render_picks_performance_tab()
    
    # === 策略优化 Tab (新增) ===
    with tab_optimizer:
        render_strategy_optimizer_tab()

    
    # === 历史复盘 Tab (新增) ===
    with tab_review:
        render_historical_review()
    
    # === 风控计算器 Tab ===
    with tab_risk:
        st.subheader("🛡️ 仓位与风控计算器")
        st.caption("基于凯利公式和ATR计算最优仓位和止损")
        
        from backtest.risk_manager import RiskManager
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_capital = st.number_input("总资金", value=100000.0, step=10000.0)
            stock_price = st.number_input("股票价格", value=50.0, step=1.0)
        with col2:
            win_rate = st.slider("历史胜率%", 30, 80, 55) / 100
            avg_win = st.number_input("平均盈利%", value=8.0, step=1.0)
        with col3:
            avg_loss = st.number_input("平均亏损%", value=4.0, step=1.0)
            atr = st.number_input("ATR (可选)", value=2.0, step=0.5)
        
        if st.button("📊 计算仓位建议", key="calc_risk"):
            rm = RiskManager(total_capital=total_capital)
            
            # 计算建议
            rec = rm.recommend_position(
                symbol="INPUT",
                price=stock_price,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                atr=atr if atr > 0 else None
            )
            
            st.success("✅ 计算完成")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("建议股数", f"{rec['shares']} 股")
                st.metric("仓位比例", f"{rec['position_pct']:.1f}%")
            with col_b:
                st.metric("入场价", f"${rec['entry_price']:.2f}")
                st.metric("止损价", f"${rec['stop_loss']:.2f}")
            with col_c:
                st.metric("止盈价", f"${rec['take_profit']:.2f}")
                st.metric("风险回报比", f"1:{rec['risk_reward']}")
            
            # Kelly 公式解释
            with st.expander("📚 凯利公式说明"):
                kelly_raw = rm.calc_position_size_kelly(stock_price, win_rate, avg_win, avg_loss)
                st.markdown(f"""
                **凯利公式**: f* = W - (1-W)/R
                
                - 胜率 W = {win_rate*100:.0f}%
                - 盈亏比 R = {avg_win/avg_loss:.2f}
                - 原始Kelly仓位 = {kelly_raw.get('kelly_raw', 0):.1f}%
                - 调整后仓位 (1/4 Kelly) = {kelly_raw.get('kelly_adjusted', 0):.1f}%
                
                *使用分数凯利更保守,避免过度下注*
                """)
    
    # === 单股回测 Tab ===
    with tab_single:
        st.info("在这里您可以对单只股票进行历史回测，验证策略参数的有效性。")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol_input = st.text_input("股票代码", value="NVDA", help="例如: NVDA, AAPL")
            symbol = symbol_input.upper().strip() if symbol_input else ""
        with col2:
            market = st.selectbox("市场", ["US", "CN"], index=0)
    with col3:
        initial_capital = st.number_input("初始资金", value=100000.0, step=10000.0)
    with col4:
        days = st.number_input("回测天数", value=1095, step=365, help="365天 = 1年")
        
    col5, col6 = st.columns(2)
    with col5:
        threshold = st.slider("BLUE 买入阈值", min_value=50.0, max_value=200.0, value=100.0, step=10.0)
    with col6:
        commission = st.number_input("佣金费率", value=0.001, format="%.4f")
        
    col7, col8 = st.columns(2)
    with col7:
        require_heima = st.checkbox("✅ 必须包含黑马/掘底信号", value=False, help="更严格：仅当同时出现黑马或掘底信号时才买入")
    with col8:
        require_week_blue = st.checkbox("✅ 必须包含周线BLUE共振", value=False, help="更严格：仅当周线BLUE同时也大于阈值时才买入")
        
    require_vp = st.checkbox("✅ 必须筹码形态良好", value=False, help="过滤掉获利盘极低且被筹码峰压制的假反弹")
    
    # 风控选项
    use_risk_mgmt = st.checkbox("🛡️ 启用专业风控 (ATR止损 + 动态仓位)", value=True, help="启用后，不再全仓买入。基于ATR计算仓位(单笔风险2%)，并使用移动止损。")
    
    # --- 智能推荐模块 ---
    if st.button("🚀 开始回测"):
        with st.spinner(f"正在回测 {symbol} ..."):
            try:
                # 初始化回测引擎
                backtester = SimpleBacktester(
                    symbol=symbol, 
                    market=market, 
                    initial_capital=initial_capital, 
                    days=days, 
                    commission_rate=commission,
                    blue_threshold=threshold,
                    require_heima=require_heima,
                    require_week_blue=require_week_blue,
                    require_vp_filter=require_vp,
                    use_risk_management=use_risk_mgmt
                )
                
                # 加载数据
                if not backtester.load_data():
                    st.error(f"❌ 数据加载失败: 无法获取 {symbol} 的数据。可能是网络问题或API限制，请稍后重试。")
                else:
                    # 运行回测
                    backtester.calculate_signals()
                    backtester.run_backtest()
                    
                    # 显示结果摘要
                    res = backtester.results
                    
                    st.success(f"✅ 回测完成！")
                    
                    # 显示自适应信息
                    if 'Adaptive Info' in res:
                        st.info(f"🤖 **自适应引擎已激活**: {res['Adaptive Info']}")
                    
                    # 关键指标卡片
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("总收益率", f"{res['Total Return']:.2%}", delta_color="normal")
                    m2.metric("年化收益率", f"{res['Annual Return']:.2%}")
                    m3.metric("最大回撤", f"{res['Max Drawdown']:.2%}", delta_color="inverse")
                    m4.metric("胜率", f"{res['Win Rate']:.2%}", f"{res['Total Trades']} 笔交易")
                    
                    # 资金曲线图
                    fig = backtester.plot_results(show=False)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # 交易详情表
                    if backtester.trades:
                        st.subheader("📋 交易记录 & 风控详情")
                        
                        trade_data = []
                        for t in backtester.trades:
                            trade_data.append({
                                "日期": t['date'].strftime('%Y-%m-%d'),
                                "类型": t['type'],
                                "价格": f"{t['price']:.2f}",
                                "数量": t['shares'],
                                "金额": f"{t['value']:.2f}",
                                "盈亏": f"{t.get('pnl', 0):.2f}" if 'pnl' in t else "-",
                                "交易理由": t.get('reason', '-'),
                                "止损价": f"{t.get('stop_loss', 0):.2f}" if t.get('stop_loss', 0) > 0 else "-"
                            })
                        
                        st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
                    else:
                        st.warning("在此期间未触发任何交易。")

                    # 被过滤的信号表
                    if hasattr(backtester, 'rejected_trades') and backtester.rejected_trades:
                        with st.expander("🚫 查看被过滤的信号 (诊断报告)", expanded=True):
                            st.caption("以下信号满足了基础 BLUE 阈值，但被您的高级过滤条件（周线/黑马/筹码分布）拒绝。")
                            
                            rejected_data = []
                            for r in backtester.rejected_trades:
                                rejected_data.append({
                                    "日期": r['date'].strftime('%Y-%m-%d'),
                                    "价格": f"{r['price']:.2f}",
                                    "Day BLUE": f"{r['blue']:.1f}",
                                    "Week BLUE": f"{r.get('week_blue', 0):.1f}",
                                    "拒绝原因 ❌": r['reason']
                                })
                            
                            st.dataframe(pd.DataFrame(rejected_data), use_container_width=True)
                        
            except Exception as e:
                st.error(f"回测出错: {str(e)}")

# --- Baseline 对比页面 ---

def render_baseline_comparison_page():
    """Baseline 扫描对比页面"""
    st.header("🔄 Baseline 对比 (Scan Comparison)")
    st.info("对比 Baseline 扫描方法与当前扫描方法的结果差异")
    
    from db.database import query_baseline_results, compare_scan_results, get_scanned_dates
    
    # 侧边栏设置
    with st.sidebar:
        st.subheader("📊 对比设置")
        
        # 市场选择
        market = st.radio("选择市场", ["🇺🇸 US", "🇨🇳 CN"], horizontal=True, key="cmp_market")
        market_code = "US" if "US" in market else "CN"
        
        # 获取可用日期
        dates = _cached_scanned_dates(market=market_code)
        if not dates:
            st.warning("暂无扫描数据")
            return
        
        selected_date = st.selectbox("选择日期", dates[:30], key="cmp_date")
        
        compare_btn = st.button("🔍 开始对比", type="primary", use_container_width=True)
    
    if not compare_btn:
        st.markdown("""
        ### 使用说明
        
        1. 在左侧选择 **市场** 和 **日期**
        2. 点击 **开始对比** 按钮
        3. 查看两种扫描方法的结果差异
        
        #### Baseline vs 当前方法
        - **Baseline**: 原始的 BLUE 信号扫描算法
        - **当前方法**: 包含更多过滤条件的优化版本
        """)
        return
    
    with st.spinner("正在对比数据..."):
        comparison = compare_scan_results(selected_date, market_code)
        baseline_results = query_baseline_results(scan_date=selected_date, market=market_code, limit=200)
    
    # 显示统计摘要
    st.markdown("---")
    st.markdown("### 📊 对比统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Baseline 信号数", comparison['baseline_count'])
    with col2:
        st.metric("当前方法信号数", comparison['regular_count'])
    with col3:
        st.metric("共同发现", len(comparison['both']))
    with col4:
        overlap = 0
        if comparison['baseline_count'] > 0:
            overlap = len(comparison['both']) / comparison['baseline_count'] * 100
        st.metric("重叠率", f"{overlap:.0f}%")
    
    # 三列布局显示差异
    st.markdown("---")
    st.markdown("### 📋 详细对比")
    
    tab1, tab2, tab3 = st.tabs(["🟢 共同发现", "🔵 仅 Baseline", "🟠 仅当前方法"])
    
    with tab1:
        if comparison['both']:
            st.success(f"两种方法共同发现 {len(comparison['both'])} 只股票")
            st.write(", ".join(comparison['both'][:50]))
        else:
            st.info("没有共同发现的股票")
    
    with tab2:
        if comparison['baseline_only']:
            st.info(f"Baseline 独有 {len(comparison['baseline_only'])} 只股票（当前方法未发现）")
            st.write(", ".join(comparison['baseline_only'][:50]))
        else:
            st.success("Baseline 没有独有的发现")
    
    with tab3:
        if comparison['regular_only']:
            st.info(f"当前方法独有 {len(comparison['regular_only'])} 只股票（Baseline 未发现）")
            st.write(", ".join(comparison['regular_only'][:50]))
        else:
            st.success("当前方法没有独有的发现")
    
    # Baseline 详细结果
    if baseline_results:
        st.markdown("---")
        st.markdown("### 📈 Baseline 详细结果")
        
        df = pd.DataFrame(baseline_results)
        display_cols = ['symbol', 'company_name', 'price', 'latest_day_blue', 'latest_week_blue', 'scan_time']
        available_cols = [c for c in display_cols if c in df.columns]
        
        if available_cols:
            display_df = df[available_cols].copy()
            display_df.columns = ['代码', '名称', '价格', 'Day BLUE', 'Week BLUE', '扫描时段'][:len(available_cols)]
            st.dataframe(display_df, hide_index=True, use_container_width=True)


# --- ML Lab 页面 (新增) ---

def render_ml_lab_page():
    """机器学习实验室 - 统计ML、深度学习、LLM"""
    st.header("🤖 ML 实验室 (Machine Learning Lab)")
    
    # 检查依赖
    from ml.statistical_models import check_ml_dependencies, get_available_models
    deps = check_ml_dependencies()
    
    # 显示依赖状态
    with st.expander("📦 ML 依赖状态", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅" if deps['sklearn'] else "❌"
            st.write(f"{status} scikit-learn")
        with col2:
            status = "✅" if deps['xgboost'] else "❌"
            st.write(f"{status} XGBoost")
        with col3:
            status = "✅" if deps['lightgbm'] else "❌"
            st.write(f"{status} LightGBM")
        
        if not all(deps.values()):
            st.code("pip install scikit-learn xgboost lightgbm", language="bash")
    
    # 四个 Tab (新增 AutoML 和 Ensemble)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 统计ML", "🧠 深度学习", "💬 LLM智能", "🔧 AutoML/集成"])
    
    with tab1:
        st.subheader("统计机器学习")
        st.info("使用 XGBoost, LightGBM, Random Forest 等模型预测信号成功率")
        
        available_models = get_available_models()
        if not available_models:
            st.error("未安装任何 ML 依赖。请运行: `pip install scikit-learn xgboost lightgbm`")
            return
        
        # 参数设置
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.selectbox("选择模型", available_models, help="XGBoost 通常表现最好")
        with col2:
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
        with col3:
            forward_days = st.selectbox("目标收益周期", [5, 10, 20], index=1, help="预测 N 天后的收益")
        
        # 数据范围
        st.markdown("#### 📅 训练数据范围")
        col4, col5, col6 = st.columns(3)
        with col4:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            train_start = st.date_input("开始日期", value=start_date)
        with col5:
            train_end = st.date_input("结束日期", value=end_date)
        with col6:
            min_blue = st.slider("最低 BLUE 阈值", 50, 150, 80, 10)
        
        # 训练按钮
        if st.button("🚀 开始训练", type="primary", use_container_width=True):
            with st.spinner("正在准备数据并训练模型..."):
                try:
                    # 1. 优先从缓存加载数据
                    from db.database import query_signal_performance, get_performance_stats
                    from ml.feature_engineering import prepare_training_data
                    from ml.statistical_models import SignalClassifier
                    
                    st.text("📊 正在从缓存加载历史信号数据...")
                    
                    # 尝试从缓存读取
                    cached_data = query_signal_performance(
                        start_date=train_start.strftime('%Y-%m-%d'),
                        end_date=train_end.strftime('%Y-%m-%d'),
                        market='US',
                        limit=1000
                    )
                    
                    if len(cached_data) >= 30:
                        st.text(f"✅ 从缓存加载了 {len(cached_data)} 条性能数据")
                        
                        # 转换为训练格式
                        ret_col = f'return_{forward_days}d'
                        valid_data = [d for d in cached_data if d.get(ret_col) is not None and d.get('blue_daily') is not None]
                        
                        import pandas as pd
                        X = pd.DataFrame([{
                            'blue_daily': d.get('blue_daily', 0),
                            'price': d.get('price', 0),
                        } for d in valid_data])
                        
                        y = pd.Series([1 if d[ret_col] > 0 else 0 for d in valid_data])
                        
                    else:
                        st.warning(f"⚠️ 缓存数据不足 ({len(cached_data)} 条)，尝试实时计算...")
                        
                        # 回退到实时计算
                        from services.backtest_service import run_signal_backtest
                        
                        result = run_signal_backtest(
                            start_date=train_start.strftime('%Y-%m-%d'),
                            end_date=train_end.strftime('%Y-%m-%d'),
                            market='US',
                            min_blue=min_blue,
                            forward_days=forward_days,
                            limit=500
                        )
                        
                        signals = result.get('signals', [])
                        if len(signals) < 30:
                            st.error(f"❌ 数据不足！仅找到 {len(signals)} 个信号")
                            st.info("💡 运行: `python scripts/compute_performance.py --limit 200` 预计算性能数据")
                            return
                        
                        X, y = prepare_training_data(signals, forward_days, 'binary')
                    
                    if X.empty or len(y) < 30:
                        st.error("❌ 特征准备失败，可能是收益数据不足")
                        return
                    
                    st.text(f"✅ 特征矩阵: {X.shape[0]} 样本, {X.shape[1]} 特征")
                    
                    # 3. 训练模型
                    st.text(f"🧠 正在训练 {model_type} 模型...")
                    classifier = SignalClassifier(model_type=model_type)
                    metrics = classifier.train(X, y, test_size=test_size)
                    
                    st.success("✅ 模型训练完成!")
                    
                    # 4. 显示结果
                    st.markdown("---")
                    st.subheader("📈 模型性能")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        acc = metrics.get('accuracy', 0) * 100
                        st.metric("准确率 (Accuracy)", f"{acc:.1f}%", 
                                 delta="好" if acc > 55 else "需改进")
                    with m2:
                        prec = metrics.get('precision', 0) * 100
                        st.metric("精确率 (Precision)", f"{prec:.1f}%")
                    with m3:
                        rec = metrics.get('recall', 0) * 100
                        st.metric("召回率 (Recall)", f"{rec:.1f}%")
                    with m4:
                        f1 = metrics.get('f1', 0) * 100
                        st.metric("F1 Score", f"{f1:.1f}%")
                    
                    # 5. 特征重要性
                    importance_df = classifier.get_feature_importance_df()
                    if not importance_df.empty:
                        st.markdown("---")
                        st.subheader("📊 特征重要性")
                        
                        import plotly.express as px
                        fig = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Feature Importance",
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 6. 模型解释
                    st.markdown("---")
                    st.subheader("💡 模型解读")
                    
                    if acc > 55:
                        st.success(f"""
                        **模型表现良好!** 准确率 {acc:.1f}% 高于随机猜测 (50%)。
                        
                        - 该模型可以作为信号筛选的辅助参考
                        - 高 BLUE 值的信号有更高的盈利概率
                        - 建议结合其他技术指标使用
                        """)
                    else:
                        st.warning(f"""
                        **模型准确率较低** ({acc:.1f}%)，可能原因：
                        
                        - 训练数据量不足 (当前: {len(signals)} 个信号)
                        - 特征与目标的相关性不强
                        - 市场噪音较大，难以预测
                        
                        💡 **建议**: 积累更多历史数据后重新训练
                        """)
                    
                except ImportError as e:
                    st.error(f"❌ 缺少依赖: {e}")
                    st.code("pip install scikit-learn xgboost lightgbm", language="bash")
                except Exception as e:
                    st.error(f"❌ 训练出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab2:
        st.subheader("深度学习 🧠")
        st.info("使用 LSTM/GRU 时间序列模型进行价格预测")
        
        from ml.deep_learning import check_torch_available
        
        if not check_torch_available():
            st.error("❌ PyTorch 未安装")
            st.code("pip install torch", language="bash")
            return
        
        st.success("✅ PyTorch 已安装")
        
        # 参数设置
        col1, col2, col3 = st.columns(3)
        with col1:
            dl_symbol = st.text_input("股票代码", value="AAPL", help="例如: AAPL, NVDA, TSLA")
        with col2:
            dl_model = st.selectbox("模型类型", ["LSTM", "GRU"], help="LSTM 更稳定, GRU 更快")
        with col3:
            dl_days = st.slider("训练数据天数", 50, 200, 100, 10)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            seq_length = st.slider("序列长度", 10, 50, 20, 5, help="回看多少天预测未来")
        with col5:
            dl_epochs = st.slider("训练轮数", 20, 200, 50, 10)
        with col6:
            hidden_size = st.selectbox("隐藏层大小", [32, 64, 128], index=1)
        
        if st.button("🚀 开始训练", type="primary", key="dl_train"):
            with st.spinner(f"正在训练 {dl_model} 模型..."):
                try:
                    from ml.deep_learning import train_price_predictor
                    
                    result = train_price_predictor(
                        symbol=dl_symbol.upper(),
                        days=dl_days,
                        seq_length=seq_length,
                        epochs=dl_epochs,
                        model_type=dl_model
                    )
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                        return
                    
                    st.success("✅ 训练完成!")
                    
                    # 显示指标
                    st.markdown("---")
                    st.subheader("📈 预测性能")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("MAE (平均绝对误差)", f"${result['mae']:.2f}")
                    with m2:
                        st.metric("RMSE (均方根误差)", f"${result['rmse']:.2f}")
                    with m3:
                        acc = result['direction_accuracy'] * 100
                        st.metric("方向准确率", f"{acc:.1f}%", 
                                 delta="好" if acc > 55 else "待改进")
                    with m4:
                        st.metric("验证损失", f"{result['val_loss']:.6f}")
                    
                    # 训练曲线
                    st.markdown("---")
                    st.subheader("📉 训练损失曲线")
                    
                    chart_data = result.get('chart_data', {})
                    if chart_data:
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=chart_data['epochs'],
                            y=chart_data['train_loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#58a6ff', width=2)
                        ))
                        if chart_data.get('val_loss'):
                            fig.add_trace(go.Scatter(
                                x=chart_data['epochs'],
                                y=chart_data['val_loss'],
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='#f0883e', width=2, dash='dot')
                            ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=300,
                            xaxis_title="Epoch",
                            yaxis_title="Loss (MSE)",
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 预测 vs 实际
                    st.markdown("---")
                    st.subheader("🎯 预测 vs 实际 (最近10天)")
                    
                    pred_df = pd.DataFrame({
                        '实际价格': result.get('actuals', []),
                        '预测价格': result.get('predictions', [])
                    })
                    st.dataframe(pred_df.style.format("${:.2f}"), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ 训练出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab3:
        st.subheader("LLM 智能分析 💬")
        st.info("使用大语言模型进行市场分析和自然语言查询")
        
        from ml.llm_intelligence import check_llm_available, LLMAnalyzer
        
        # 检查 API 状态
        llm_status = check_llm_available()
        
        col1, col2 = st.columns(2)
        with col1:
            status = "✅" if llm_status['openai'] else "❌"
            st.write(f"{status} OpenAI SDK")
        with col2:
            status = "✅" if llm_status['anthropic'] else "❌"
            st.write(f"{status} Anthropic SDK")
        
        # API Key 状态
        openai_key = os.environ.get('OPENAI_API_KEY', '')
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
        
        if not openai_key and not anthropic_key:
            st.warning("⚠️ 未配置 API Key。请设置 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY` 环境变量。")
            st.code("export OPENAI_API_KEY='your-api-key'", language="bash")
            
            # 允许临时输入
            with st.expander("🔑 临时输入 API Key"):
                temp_key = st.text_input("OpenAI API Key", type="password", key="temp_openai")
                if temp_key:
                    os.environ['OPENAI_API_KEY'] = temp_key
                    st.success("✅ API Key 已设置 (仅本次会话有效)")
                    st.rerun()
            return
        
        # 选择提供商
        provider = "openai" if openai_key else "anthropic"
        st.success(f"✅ 已配置 {provider.upper()} API")
        
        # 三个子功能
        llm_tab1, llm_tab2, llm_tab3 = st.tabs(["💬 AI 问答", "📊 情感分析", "📝 市场报告"])
        
        with llm_tab1:
            st.markdown("### 💬 AI 问答助手")
            st.caption("问我任何关于量化交易、技术指标的问题")
            
            # 聊天历史
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # 显示历史
            for msg in st.session_state.chat_history[-6:]:  # 最近 6 条
                with st.chat_message(msg['role']):
                    st.write(msg['content'])
            
            # 用户输入
            user_input = st.chat_input("输入你的问题...")
            
            if user_input:
                # 添加用户消息
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.chat_message("user"):
                    st.write(user_input)
                
                # AI 回复
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        analyzer = LLMAnalyzer(provider)
                        response = analyzer.natural_query(user_input)
                        st.write(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with llm_tab2:
            st.markdown("### 📊 新闻情感分析")
            st.caption("分析财经新闻或社交媒体情感")
            
            sample_text = st.text_area(
                "输入文本",
                placeholder="粘贴新闻标题、推文或财经评论...",
                height=100
            )
            
            if st.button("🔍 分析情感", key="sentiment_btn"):
                if sample_text:
                    with st.spinner("分析中..."):
                        analyzer = LLMAnalyzer(provider)
                        result = analyzer.analyze_sentiment(sample_text)
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            # 显示结果
                            sentiment = result.get('sentiment', 'neutral')
                            confidence = result.get('confidence', 0)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                emoji = "🟢" if sentiment == "bullish" else ("🔴" if sentiment == "bearish" else "⚪")
                                st.metric("情感", f"{emoji} {sentiment.upper()}")
                            with col2:
                                st.metric("置信度", f"{confidence:.0%}")
                            
                            st.markdown("**要点:**")
                            for point in result.get('key_points', []):
                                st.write(f"• {point}")
                            
                            st.markdown(f"**分析:** {result.get('reasoning', '')}")
                else:
                    st.warning("请输入文本")
        
        with llm_tab3:
            st.markdown("### 📝 AI 市场报告")
            st.caption("基于当日信号自动生成市场分析报告")
            
            if st.button("📄 生成报告", key="report_btn"):
                with st.spinner("正在生成报告..."):
                    # 获取今日信号
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')
                    signals = query_scan_results(scan_date=today, market='US', limit=20)
                    
                    analyzer = LLMAnalyzer(provider)
                    report = analyzer.generate_market_report(signals)
                    
                    st.markdown(report)
    
    with tab4:
        st.subheader("🔧 AutoML & 模型集成")
        st.info("自动化模型选择和多模型融合")
        
        automl_tab1, automl_tab2 = st.tabs(["🤖 AutoML", "🔗 集成预测"])
        
        with automl_tab1:
            st.markdown("### 自动模型选择")
            st.caption("自动训练多种模型并选择最优")
            
            col1, col2 = st.columns(2)
            with col1:
                automl_market = st.selectbox("市场", ["US", "CN"], key="automl_market")
            with col2:
                cv_folds = st.slider("交叉验证折数", 3, 10, 5, key="cv_folds")
            
            if st.button("🚀 运行 AutoML", key="run_automl"):
                with st.spinner("正在训练多个模型..."):
                    try:
                        from ml.ensemble import AutoML
                        from db.database import query_scan_results
                        from ml.feature_engineering import prepare_training_data
                        
                        # 获取数据
                        signals = query_scan_results(market=automl_market, limit=300)
                        if not signals or len(signals) < 50:
                            st.warning("数据不足")
                        else:
                            # 准备特征
                            X_list = []
                            y_list = []
                            for s in signals:
                                if s.get('blue_daily') is not None:
                                    X_list.append({
                                        'blue_daily': s.get('blue_daily', 0) or 0,
                                        'blue_weekly': s.get('blue_weekly', 0) or 0,
                                        'adx': s.get('adx', 0) or 0,
                                        'volatility': s.get('volatility', 0) or 0,
                                    })
                                    # 简化标签
                                    y_list.append(1 if (s.get('blue_daily', 0) or 0) > 100 else 0)
                            
                            X = pd.DataFrame(X_list).fillna(0)
                            y = np.array(y_list)
                            
                            if len(X) < 30:
                                st.warning("特征数据不足")
                            else:
                                automl = AutoML(market=automl_market)
                                result = automl.auto_train(X.values, y, cv_folds=cv_folds)
                                
                                if 'error' in result:
                                    st.error(result['error'])
                                else:
                                    st.success(f"✅ 最优模型: **{result['best_model_type']}** (CV Score: {result['best_cv_score']:.4f})")
                                    
                                    # 结果表格
                                    results_df = pd.DataFrame(result['all_results'])
                                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                                    
                                    # 保存到 session
                                    st.session_state['automl_instance'] = automl
                                    st.info("💡 可在「集成预测」Tab 使用这些模型创建集成")
                                    
                    except Exception as e:
                        st.error(f"AutoML 出错: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with automl_tab2:
            st.markdown("### 模型集成预测")
            st.caption("融合多个模型的预测结果")
            
            if 'automl_instance' in st.session_state:
                automl = st.session_state['automl_instance']
                
                # 创建集成
                if st.button("创建集成", key="create_ensemble"):
                    try:
                        ensemble = automl.create_ensemble()
                        st.session_state['ensemble'] = ensemble
                        st.success("✅ 集成已创建!")
                        
                        # 显示集成摘要
                        summary = ensemble.summary()
                        st.dataframe(summary, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"创建集成失败: {e}")
                
                if 'ensemble' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 使用集成预测")
                    
                    symbol = st.text_input("输入股票代码", value="AAPL", key="ensemble_symbol")
                    
                    if st.button("预测", key="ensemble_predict"):
                        st.info("正在预测... (演示)")
                        # 这里可以接入实际预测逻辑
                        prob = np.random.uniform(0.4, 0.8)
                        st.metric("盈利概率", f"{prob:.1%}")
            else:
                st.info("请先在「AutoML」Tab 训练模型")


# --- 博主推荐追踪页面 ---

def render_external_strategies_tab():
    """📊 外部策略 - TradingView 和社区策略"""
    st.subheader("📊 外部策略库")
    st.caption("TradingView 热门策略、社区策略、博主策略")
    
    try:
        from strategies.aggregator import StrategyAggregator, StrategySource, StrategyCategory
        from strategies.implementations import list_strategies
        
        aggregator = StrategyAggregator()
        
        # TradingView 热门策略
        st.markdown("### 📈 TradingView 热门策略")
        
        tv_strategies = aggregator.tv_scraper.get_popular_strategies()
        
        if tv_strategies:
            tv_df = pd.DataFrame([
                {
                    '策略名称': s.name,
                    '类别': s.category.value if isinstance(s.category, StrategyCategory) else s.category,
                    '入场规则': s.entry_rules[:50] + '...' if len(s.entry_rules) > 50 else s.entry_rules,
                    '出场规则': s.exit_rules[:50] + '...' if len(s.exit_rules) > 50 else s.exit_rules,
                    '声称胜率': f"{s.claimed_win_rate}%",
                    '主要指标': ', '.join(s.indicators[:3])
                }
                for s in tv_strategies
            ])
            
            st.dataframe(tv_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 可回测的策略
        st.markdown("### 🧪 可回测策略")
        st.caption("这些策略已实现完整逻辑，可直接回测")
        
        impl_strategies = list_strategies()
        
        impl_df = pd.DataFrame([
            {
                '策略ID': s['id'],
                '策略名称': s['name'],
                '描述': s['description'],
                '使用指标': ', '.join(s.get('indicators', []))
            }
            for s in impl_strategies
        ])
        
        st.dataframe(impl_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 博主列表
        st.markdown("### 👤 知名博主")
        
        authors = aggregator.get_all_authors()
        
        if authors:
            author_df = pd.DataFrame([
                {
                    '博主': a.name,
                    '平台': a.platform.value if isinstance(a.platform, StrategySource) else a.platform,
                    '专长': a.specialty,
                    '粉丝数': f"{a.followers:,}" if a.followers else 'N/A',
                    '简介': a.description[:30] + '...' if len(a.description) > 30 else a.description
                }
                for a in authors
            ])
            
            st.dataframe(author_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_article_crawler_tab():
    """🔍 文章爬取与策略分析 - 自动爬取量化博客文章"""
    st.subheader("🔍 量化博客文章爬取")
    st.caption("自动爬取中英文量化博客，分析其中的策略并回测验证")
    
    # 数据源列表
    st.markdown("### 📚 数据源")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🇺🇸 英文源**")
        en_sources = [
            ("Quantocracy", "量化聚合站", "https://quantocracy.com/"),
            ("Alpha Architect", "因子研究", "https://alphaarchitect.com/blog/"),
            ("Quantpedia", "策略库", "https://quantpedia.com/blog/"),
            ("SSRN Finance", "学术论文", "https://papers.ssrn.com/"),
            ("QuantStart", "量化教程", "https://www.quantstart.com/"),
        ]
        for name, cat, url in en_sources:
            st.markdown(f"• **{name}** - {cat}")
    
    with col2:
        st.markdown("**🇨🇳 中文源**")
        cn_sources = [
            ("雪球热帖", "社区热门", "https://xueqiu.com/"),
            ("聚宽社区", "量化策略", "https://www.joinquant.com/"),
            ("米筐研究", "量化策略", "https://www.ricequant.com/"),
            ("同花顺量化", "量化资讯", "https://quant.10jqka.com.cn/"),
        ]
        for name, cat, url in cn_sources:
            st.markdown(f"• **{name}** - {cat}")
    
    st.divider()
    
    # 爬取控制
    st.markdown("### 🚀 爬取文章")
    
    col_fetch1, col_fetch2 = st.columns(2)
    
    with col_fetch1:
        fetch_lang = st.radio("选择语言", ["全部", "英文", "中文"], horizontal=True)
    
    with col_fetch2:
        use_llm = st.checkbox("使用 LLM 分析策略", value=False, 
                              help="使用 GPT-4 提取更精准的策略，需要 OPENAI_API_KEY")
    
    if st.button("🔍 开始爬取", type="primary"):
        try:
            from services.blogger_tracker import (
                ArticleFetcher, StrategyExtractor, StrategyBacktester,
                BloggerTrackerDB
            )
            
            with st.spinner("正在爬取文章..."):
                fetcher = ArticleFetcher()
                results = fetcher.fetch_all(save=True)
                
                en_count = len(results.get('en', []))
                cn_count = len(results.get('cn', []))
                
                st.success(f"✅ 爬取完成! 英文: {en_count} 篇, 中文: {cn_count} 篇")
            
            # 分析策略
            if en_count + cn_count > 0:
                with st.spinner("正在分析策略..."):
                    db = BloggerTrackerDB()
                    extractor = StrategyExtractor()
                    
                    articles = db.get_recent_articles(days=1)
                    strategies_found = 0
                    
                    progress = st.progress(0)
                    for i, article in enumerate(articles):
                        if use_llm:
                            strategy = extractor.extract_strategy_with_llm(article)
                        else:
                            strategy = extractor.extract_strategy_rule_based(article)
                        
                        if strategy:
                            db.save_strategy(strategy)
                            strategies_found += 1
                        
                        progress.progress((i + 1) / len(articles))
                    
                    progress.empty()
                    st.success(f"✅ 分析完成! 提取了 {strategies_found} 个策略")
                    
                    # 回测
                    if strategies_found > 0:
                        with st.spinner("正在回测策略..."):
                            backtester = StrategyBacktester()
                            strategies_list = db.get_strategies_with_backtests()
                            
                            backtest_count = 0
                            for strategy in strategies_list:
                                if strategy.get('total_return') is None:
                                    result = backtester.backtest_extracted_strategy(strategy)
                                    if result:
                                        db.save_backtest(result)
                                        backtest_count += 1
                            
                            st.success(f"✅ 回测完成! 回测了 {backtest_count} 个策略")
        
        except ImportError as e:
            st.error(f"需要安装依赖: {e}")
            st.code("pip install beautifulsoup4 lxml")
        except Exception as e:
            st.error(f"爬取失败: {e}")
    
    st.divider()
    
    # 显示已爬取的文章
    st.markdown("### 📰 最新文章")
    
    try:
        from services.blogger_tracker import BloggerTrackerDB
        
        db = BloggerTrackerDB()
        articles = db.get_recent_articles(days=7)
        
        if articles:
            article_df = pd.DataFrame([
                {
                    '来源': a['source'],
                    '标题': a['title'][:50] + '...' if len(a['title']) > 50 else a['title'],
                    '作者': a['author'],
                    '类别': a['category'],
                    '语言': '🇨🇳' if a['language'] == 'cn' else '🇺🇸',
                    '日期': a['publish_date'],
                    '已分析': '✅' if a.get('analyzed') else '❌'
                }
                for a in articles[:30]
            ])
            
            st.dataframe(article_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无文章，请点击「开始爬取」")
    except Exception as e:
        st.warning(f"加载文章失败: {e}")
    
    st.divider()
    
    # 策略排行榜
    st.markdown("### 🏆 策略排行榜")
    st.caption("根据回测结果排序，展示最有效的策略")
    
    try:
        from services.blogger_tracker import BloggerTrackerDB
        
        db = BloggerTrackerDB()
        strategies = db.get_strategies_with_backtests()
        
        # 只显示有回测结果的
        strategies_with_bt = [s for s in strategies if s.get('total_return') is not None]
        
        if strategies_with_bt:
            # 按收益排序
            strategies_with_bt.sort(key=lambda x: x.get('sharpe_ratio', 0) or 0, reverse=True)
            
            strat_df = pd.DataFrame([
                {
                    '策略名称': s['strategy_name'][:40],
                    '类型': s['strategy_type'],
                    '来源文章': s.get('article_title', '')[:30] if s.get('article_title') else '-',
                    '总收益': f"{s['total_return']:.1f}%" if s.get('total_return') else '-',
                    'Sharpe': f"{s['sharpe_ratio']:.2f}" if s.get('sharpe_ratio') else '-',
                    '最大回撤': f"{s['max_drawdown']:.1f}%" if s.get('max_drawdown') else '-',
                    '胜率': f"{s['win_rate']:.0f}%" if s.get('win_rate') else '-',
                    '有效': '✅' if s.get('is_profitable') else '❌'
                }
                for s in strategies_with_bt[:20]
            ])
            
            st.dataframe(
                strat_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "策略名称": st.column_config.TextColumn("策略名称", width="large"),
                    "类型": st.column_config.TextColumn("类型", width="small"),
                    "来源文章": st.column_config.TextColumn("来源", width="medium"),
                    "总收益": st.column_config.TextColumn("收益", width="small"),
                    "Sharpe": st.column_config.TextColumn("Sharpe", width="small"),
                    "最大回撤": st.column_config.TextColumn("回撤", width="small"),
                    "胜率": st.column_config.TextColumn("胜率", width="small"),
                    "有效": st.column_config.TextColumn("有效", width="small"),
                }
            )
            
            # 统计
            profitable_count = sum(1 for s in strategies_with_bt if s.get('is_profitable'))
            st.info(f"📊 统计: {len(strategies_with_bt)} 个策略已回测, {profitable_count} 个盈利 ({profitable_count/len(strategies_with_bt)*100:.0f}%)")
        else:
            st.info("暂无回测结果，请先爬取并分析文章")
    except Exception as e:
        st.warning(f"加载策略失败: {e}")


def render_strategy_backtest_tab():
    """🧪 策略回测 - 回测外部策略"""
    st.subheader("🧪 外部策略回测")
    st.caption("选择策略和股票，验证策略有效性")
    
    try:
        from strategies.implementations import (
            list_strategies, backtest_external_strategy, get_strategy
        )
        
        # 策略选择
        strategies = list_strategies()
        strategy_options = {s['name']: s['id'] for s in strategies}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_name = st.selectbox("选择策略", list(strategy_options.keys()))
            selected_id = strategy_options.get(selected_name)
        
        with col2:
            symbol = st.text_input("股票代码", value="NVDA").upper().strip()
        
        with col3:
            days = st.selectbox("回测周期", [90, 180, 365, 730], index=2)
        
        # 显示策略详情
        strategy_info = next((s for s in strategies if s['id'] == selected_id), None)
        if strategy_info:
            st.info(f"**{strategy_info['name']}**: {strategy_info['description']}")
            st.caption(f"使用指标: {', '.join(strategy_info.get('indicators', []))}")
        
        # 运行回测
        if st.button("🚀 运行回测", type="primary", key="run_ext_backtest"):
            with st.spinner(f"正在回测 {selected_name} on {symbol}..."):
                result = backtest_external_strategy(selected_id, symbol, days=days)
                
                if 'error' in result:
                    st.error(result['error'])
                elif result.get('total_signals', 0) == 0:
                    st.warning("该策略在此期间未产生任何信号")
                else:
                    st.success("✅ 回测完成!")
                    
                    # 显示结果
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("总信号数", result.get('total_signals', 0))
                    m2.metric("完成交易", result.get('completed_trades', 0))
                    m3.metric("胜率", f"{result.get('win_rate', 0)}%")
                    m4.metric("总收益", f"{result.get('total_return', 0)}%")
                    
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("平均收益", f"{result.get('avg_return', 0)}%")
                    m6.metric("最大盈利", f"{result.get('max_gain', 0)}%")
                    m7.metric("最大亏损", f"{result.get('max_loss', 0)}%")
                    m8.metric("Sharpe", result.get('sharpe', 0))
        
        st.divider()
        
        # 批量对比
        st.markdown("### 📊 策略对比")
        st.caption("比较多个策略在同一股票上的表现")
        
        compare_symbol = st.text_input("对比股票", value="AAPL", key="compare_symbol").upper()
        
        if st.button("📊 对比所有策略", key="compare_all_strategies"):
            with st.spinner("正在对比..."):
                results = []
                
                for s in strategies:
                    try:
                        r = backtest_external_strategy(s['id'], compare_symbol, days=365)
                        if 'error' not in r and r.get('completed_trades', 0) > 0:
                            results.append({
                                '策略': s['name'],
                                '信号数': r.get('total_signals', 0),
                                '交易数': r.get('completed_trades', 0),
                                '胜率': f"{r.get('win_rate', 0)}%",
                                '总收益': f"{r.get('total_return', 0)}%",
                                'Sharpe': r.get('sharpe', 0)
                            })
                    except:
                        pass
                
                if results:
                    compare_df = pd.DataFrame(results)
                    # 按总收益排序
                    compare_df['_sort'] = compare_df['总收益'].str.replace('%', '').astype(float)
                    compare_df = compare_df.sort_values('_sort', ascending=False).drop('_sort', axis=1)
                    
                    st.dataframe(compare_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("没有足够数据进行对比")
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_blogger_page():
    """📢 博主推荐追踪页面"""
    st.header("📢 博主推荐追踪")
    st.caption("追踪知名博主的股票推荐，计算收益表现")
    
    from db.database import (
        init_blogger_tables, get_all_bloggers, add_blogger, delete_blogger,
        get_recommendations, add_recommendation, delete_recommendation, get_blogger_stats
    )
    from services.blogger_service import (
        get_recommendations_with_returns,
        get_blogger_performance,
        collect_social_kol_recommendations,
    )
    
    # 确保表存在
    init_blogger_tables()
    
    tab_bloggers, tab_recs, tab_perf, tab_eval, tab_external, tab_backtest, tab_crawler = st.tabs([
        "👤 博主管理",
        "📝 推荐记录", 
        "🏆 业绩排行",
        "🎯 喊单评估",
        "📊 外部策略",
        "🧪 策略回测",
        "🔍 文章爬取"
    ])
    
    # === Tab 1: 博主管理 ===
    with tab_bloggers:
        st.subheader("博主列表")
        
        bloggers = get_all_bloggers()
        
        if bloggers:
            for b in bloggers:
                with st.expander(f"**{b['name']}** ({b.get('platform', 'N/A')})"):
                    st.write(f"专长: {b.get('specialty', 'N/A')}")
                    st.write(f"主页: {b.get('url', 'N/A')}")
                    if is_admin():
                        if st.button(f"🗑️ 删除", key=f"del_blogger_{b['id']}"):
                            delete_blogger(b['id'])
                            st.success("已删除")
                            st.rerun()
        else:
            st.info("暂无博主，请添加")
        
        st.divider()
        
        if is_admin():
            st.subheader("➕ 添加博主")
            with st.form("add_blogger_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_name = st.text_input("博主名称*", placeholder="如：唐朝")
                    new_platform = st.selectbox("平台", ["雪球", "微博", "抖音", "Twitter", "YouTube", "其他"])
                with col2:
                    new_specialty = st.selectbox("专长", ["A股", "美股", "港股", "混合"])
                    new_url = st.text_input("主页链接", placeholder="https://...")
                
                if st.form_submit_button("添加博主", type="primary"):
                    if new_name:
                        add_blogger(new_name, platform=new_platform, specialty=new_specialty, url=new_url)
                        st.success(f"✅ 已添加博主: {new_name}")
                        st.rerun()
                    else:
                        st.error("请输入博主名称")
    
    # === Tab 2: 推荐记录 ===
    with tab_recs:
        st.subheader("推荐记录")
        
        bloggers = get_all_bloggers()
        
        if not bloggers:
            st.warning("请先添加博主")
        else:
            # 筛选
            all_recs_for_tags = get_recommendations(limit=1000)
            tag_options = sorted({(x.get("portfolio_tag") or "").strip() for x in all_recs_for_tags if (x.get("portfolio_tag") or "").strip()})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                filter_blogger = st.selectbox(
                    "选择博主",
                    options=[None] + [b['id'] for b in bloggers],
                    format_func=lambda x: "全部" if x is None else next((b['name'] for b in bloggers if b['id'] == x), x)
                )
            with col2:
                filter_market = st.selectbox("市场", ["全部", "CN", "US"])
            with col3:
                filter_tag = st.selectbox("组合标签", ["全部"] + tag_options)
            with col4:
                horizon_days = st.slider("评估周期(天)", min_value=3, max_value=90, value=10, step=1, key="blogger_rec_horizon")
            
            # 获取并显示推荐
            recs = get_recommendations_with_returns(
                blogger_id=filter_blogger,
                market=None if filter_market == "全部" else filter_market,
                portfolio_tag=None if filter_tag == "全部" else filter_tag,
                limit=80,
                horizon_days=horizon_days,
            )
            
            if recs:
                rec_df = pd.DataFrame(recs)
                display_cols = [
                    'blogger_name', 'portfolio_tag', 'ticker', 'rec_date', 'rec_type',
                    'rec_price', 'current_price', 'return_pct', 'horizon_return_pct',
                    'directional_return_pct', 'direction_ok', 'mfe_pct', 'mae_pct',
                    'target_hit', 'stop_hit', 'days_held'
                ]
                display_cols = [c for c in display_cols if c in rec_df.columns]
                
                st.dataframe(
                    rec_df[display_cols],
                    column_config={
                        'blogger_name': '博主',
                        'portfolio_tag': '组合标签',
                        'ticker': '股票',
                        'rec_date': '推荐日期',
                        'rec_type': '类型',
                        'rec_price': '推荐价',
                        'current_price': '现价',
                        'return_pct': st.column_config.NumberColumn('收益%', format="%.2f%%"),
                        'horizon_return_pct': st.column_config.NumberColumn(f'{horizon_days}天收益%', format="%.2f%%"),
                        'directional_return_pct': st.column_config.NumberColumn(f'{horizon_days}天方向收益%', format="%.2f%%"),
                        'direction_ok': st.column_config.CheckboxColumn('方向命中'),
                        'mfe_pct': st.column_config.NumberColumn('最大有利%', format="%.2f%%"),
                        'mae_pct': st.column_config.NumberColumn('最大不利%', format="%.2f%%"),
                        'target_hit': st.column_config.CheckboxColumn('到目标价'),
                        'stop_hit': st.column_config.CheckboxColumn('触发止损'),
                        'days_held': '持有天数'
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("暂无推荐记录")
            
            st.divider()
            
            # 添加推荐
            if is_admin():
                st.subheader("➕ 添加推荐")
                with st.form("add_rec_form"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        rec_blogger = st.selectbox(
                            "博主*",
                            options=[b['id'] for b in bloggers],
                            format_func=lambda x: next((b['name'] for b in bloggers if b['id'] == x), x)
                        )
                        rec_ticker = st.text_input("股票代码*", placeholder="如: 600519 或 AAPL")
                    with col2:
                        rec_market = st.selectbox("市场", ["CN", "US"])
                        rec_date = st.date_input("推荐日期", value=datetime.now())
                    with col3:
                        rec_type = st.selectbox("类型", ["BUY", "SELL", "HOLD"])
                        rec_price = st.number_input("推荐价格 (可选)", min_value=0.0, step=0.01)
                    with col4:
                        rec_tag = st.text_input("组合标签", placeholder="如: X博主-高胜率组合")
                        rec_source_url = st.text_input("来源链接", placeholder="https://...")
                    
                    rec_notes = st.text_area("推荐理由", height=80)
                    
                    if st.form_submit_button("添加推荐", type="primary"):
                        if rec_ticker and rec_blogger:
                            add_recommendation(
                                blogger_id=rec_blogger,
                                ticker=rec_ticker,
                                market=rec_market,
                                rec_date=rec_date.strftime('%Y-%m-%d'),
                                rec_price=rec_price if rec_price > 0 else None,
                                rec_type=rec_type,
                                portfolio_tag=(rec_tag.strip() if rec_tag else None),
                                notes=rec_notes,
                                source_url=(rec_source_url.strip() if rec_source_url else None),
                            )
                            st.success(f"✅ 已添加推荐: {rec_ticker}")
                            st.rerun()
                        else:
                            st.error("请填写必填项")
    
    # === Tab 3: 业绩排行 ===
    with tab_perf:
        st.subheader("🏆 博主业绩排行")
        
        if st.button("🔄 刷新统计"):
            st.cache_data.clear()
        
        all_recs_for_tags = get_recommendations(limit=1000)
        tag_options = sorted({(x.get("portfolio_tag") or "").strip() for x in all_recs_for_tags if (x.get("portfolio_tag") or "").strip()})
        p1, p2 = st.columns(2)
        with p1:
            perf_horizon = st.slider("排行评估周期(天)", min_value=3, max_value=90, value=10, step=1, key="blogger_perf_horizon")
        with p2:
            perf_tag = st.selectbox("组合标签过滤", ["全部"] + tag_options, key="blogger_perf_tag")

        perf = get_blogger_performance(
            horizon_days=perf_horizon,
            portfolio_tag=None if perf_tag == "全部" else perf_tag,
        )
        
        if perf:
            perf_df = pd.DataFrame(perf)
            
            # 高亮显示
            st.dataframe(
                perf_df[['name', 'platform', 'rec_count', 'win_rate', 'avg_return', 'avg_directional_return', 'total_return']],
                column_config={
                    'name': '博主',
                    'platform': '平台',
                    'rec_count': '推荐数',
                    'win_rate': st.column_config.NumberColumn('胜率%', format="%.1f%%"),
                    'avg_return': st.column_config.NumberColumn('平均收益%', format="%.2f%%"),
                    'avg_directional_return': st.column_config.NumberColumn('方向收益%', format="%.2f%%"),
                    'total_return': st.column_config.NumberColumn('累计收益%', format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # 胜率图表
            if len(perf_df) > 0 and perf_df['rec_count'].sum() > 0:
                import plotly.express as px
                fig = px.bar(
                    perf_df[perf_df['rec_count'] > 0],
                    x='name', y='avg_return',
                    title="博主平均收益率排名",
                    color='win_rate',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无数据，请先添加博主和推荐记录")
    
    # === Tab 4: 喊单评估 ===
    with tab_eval:
        st.subheader("🎯 买卖点有效性评估")
        st.caption("按博主/组合标签评估：方向命中率、方向收益、最大有利/不利波动、目标止损触发情况")

        with st.expander("🤖 一键抓取社交大V喊单", expanded=False):
            st.caption("按行填写：平台,名称,账号,市场(US/CN，可留空)。例如：Twitter,Roaring Kitty,TheRoaringKitty,US")
            default_kols = "\n".join([
                "Twitter,Roaring Kitty,TheRoaringKitty,US",
                "Twitter,Nancy Pelosi Tracker,pelosi_tracker,US",
                "Reddit,WallStreetBets,wallstreetbets,US",
                "雪球,雪球热榜,xueqiu,US",
                "微博,财经博主样本,sinafinance,CN",
            ])
            kol_text = st.text_area(
                "大V清单",
                value=default_kols,
                height=140,
                key="social_kol_list_text",
            )
            auto_tag = st.text_input("自动组合标签", value="AUTO_SOCIAL_KOL", key="social_kol_auto_tag")
            max_per_kol = st.slider("每位抓取上限", min_value=5, max_value=60, value=20, step=5, key="social_kol_max_per")
            if st.button("🚀 开始抓取并入库", key="run_social_kol_ingest_btn"):
                kol_configs = []
                for line in (kol_text or "").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) < 3:
                        continue
                    cfg = {
                        "platform": parts[0],
                        "name": parts[1],
                        "handle": parts[2],
                        "market": parts[3] if len(parts) >= 4 else "",
                    }
                    kol_configs.append(cfg)
                with st.spinner("抓取社交帖子并识别股票中..."):
                    ret = collect_social_kol_recommendations(
                        kol_configs=kol_configs,
                        portfolio_tag=(auto_tag or "AUTO_SOCIAL_KOL").strip(),
                        max_results_per_kol=int(max_per_kol),
                    )
                st.success(
                    f"完成：处理KOL {ret.get('processed_kols', 0)} | 新增博主 {ret.get('new_bloggers', 0)} | "
                    f"新增推荐 {ret.get('added_recommendations', 0)} | 去重跳过 {ret.get('skipped_duplicates', 0)}"
                )
                kol_stats = ret.get("kol_stats") or []
                if kol_stats:
                    st.dataframe(
                        pd.DataFrame(kol_stats),
                        column_config={
                            "name": "名称",
                            "platform": "平台",
                            "handle": "账号",
                            "posts": "抓取帖子数",
                            "ticker_hits": "识别Ticker数",
                            "added": "新增推荐",
                            "duplicates": "重复跳过",
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                if ret.get("errors"):
                    st.caption("⚠️ 部分错误: " + " | ".join(ret.get("errors", [])[:3]))

        bloggers = get_all_bloggers()
        if not bloggers:
            st.info("请先在「博主管理」添加博主，并在「推荐记录」录入喊单。")
        else:
            all_recs_for_tags = get_recommendations(limit=2000)
            tag_options = sorted({(x.get("portfolio_tag") or "").strip() for x in all_recs_for_tags if (x.get("portfolio_tag") or "").strip()})
            blogger_name_map = {b["name"]: b["id"] for b in bloggers}
            default_names = list(blogger_name_map.keys())[:3]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                eval_blogger_names = st.multiselect("博主", options=list(blogger_name_map.keys()), default=default_names)
            with c2:
                eval_market = st.selectbox("市场", ["全部", "US", "CN"], key="blogger_eval_market")
            with c3:
                eval_tag = st.selectbox("组合标签", ["全部"] + tag_options, key="blogger_eval_tag")
            with c4:
                eval_horizon = st.slider("评估周期(天)", min_value=3, max_value=90, value=10, step=1, key="blogger_eval_horizon")

            c5, c6, c7 = st.columns(3)
            with c5:
                eval_types = st.multiselect("类型", options=["BUY", "SELL", "HOLD"], default=["BUY", "SELL"], key="blogger_eval_types")
            with c6:
                eval_min_samples = st.slider("最小样本", min_value=3, max_value=100, value=8, step=1, key="blogger_eval_min_samples")
            with c7:
                eval_limit = st.number_input("每博主最大样本", min_value=20, max_value=500, value=150, step=10, key="blogger_eval_limit")

            selected_ids = [blogger_name_map[n] for n in eval_blogger_names if n in blogger_name_map]
            eval_rows = []
            for bid in selected_ids:
                recs = get_recommendations_with_returns(
                    blogger_id=bid,
                    market=None if eval_market == "全部" else eval_market,
                    portfolio_tag=None if eval_tag == "全部" else eval_tag,
                    limit=int(eval_limit),
                    horizon_days=int(eval_horizon),
                )
                for r in recs:
                    if eval_types and str(r.get("rec_type", "")).upper() not in eval_types:
                        continue
                    eval_rows.append(r)

            if not eval_rows:
                st.info("当前筛选下暂无可评估样本。")
            else:
                eval_df = pd.DataFrame(eval_rows)
                eval_df["portfolio_tag"] = eval_df.get("portfolio_tag", "").fillna("").replace("", "未分组")
                if "direction_ok" in eval_df.columns:
                    eval_df["direction_ok_num"] = eval_df["direction_ok"].map(lambda x: 1.0 if bool(x) else 0.0)
                else:
                    eval_df["direction_ok_num"] = 0.0
                for col in ["directional_return_pct", "mfe_pct", "mae_pct", "target_hit", "stop_hit"]:
                    if col not in eval_df.columns:
                        eval_df[col] = 0
                eval_df["target_hit_num"] = eval_df["target_hit"].map(lambda x: 1.0 if bool(x) else 0.0)
                eval_df["stop_hit_num"] = eval_df["stop_hit"].map(lambda x: 1.0 if bool(x) else 0.0)

                summary = (
                    eval_df.groupby(["blogger_name", "portfolio_tag"], as_index=False)
                    .agg(
                        样本数=("id", "count"),
                        方向命中率=("direction_ok_num", "mean"),
                        平均方向收益=("directional_return_pct", "mean"),
                        平均最大有利=("mfe_pct", "mean"),
                        平均最大不利=("mae_pct", "mean"),
                        目标命中率=("target_hit_num", "mean"),
                        止损触发率=("stop_hit_num", "mean"),
                    )
                )
                summary["方向命中率"] = summary["方向命中率"] * 100.0
                summary["目标命中率"] = summary["目标命中率"] * 100.0
                summary["止损触发率"] = summary["止损触发率"] * 100.0
                summary = summary[summary["样本数"] >= int(eval_min_samples)]
                summary = summary.sort_values(["方向命中率", "平均方向收益"], ascending=False)

                if not summary.empty:
                    st.markdown("**组合评估汇总**")
                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "方向命中率": st.column_config.NumberColumn("方向命中率(%)", format="%.1f%%"),
                            "平均方向收益": st.column_config.NumberColumn(f"{int(eval_horizon)}天平均方向收益(%)", format="%.2f%%"),
                            "平均最大有利": st.column_config.NumberColumn("平均最大有利(%)", format="%.2f%%"),
                            "平均最大不利": st.column_config.NumberColumn("平均最大不利(%)", format="%.2f%%"),
                            "目标命中率": st.column_config.NumberColumn("目标命中率(%)", format="%.1f%%"),
                            "止损触发率": st.column_config.NumberColumn("止损触发率(%)", format="%.1f%%"),
                        },
                    )
                else:
                    st.info("样本未达到最小阈值，调低“最小样本”后再看。")

                st.markdown("**明细（最近样本）**")
                detail_cols = [
                    "blogger_name", "portfolio_tag", "ticker", "rec_date", "rec_type",
                    "rec_price", "current_price", "horizon_return_pct",
                    "directional_return_pct", "direction_ok", "mfe_pct", "mae_pct",
                    "target_hit", "stop_hit",
                ]
                detail_cols = [c for c in detail_cols if c in eval_df.columns]
                detail_df = eval_df.sort_values("rec_date", ascending=False)[detail_cols].head(200)
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # === Tab 5: 外部策略 ===
    with tab_external:
        render_external_strategies_tab()
    
    # === Tab 6: 策略回测 ===
    with tab_backtest:
        render_strategy_backtest_tab()
    
    # === Tab 7: 文章爬取与策略分析 ===
    with tab_crawler:
        render_article_crawler_tab()


# ==================== V3 合并页面 ====================

def render_signal_center_page():
    """📈 信号中心 - 合并: 信号追踪 + 信号验证 + 健康监控"""
    st.header("📈 信号中心")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👁️ 观察追踪",
        "🩺 信号健康", 
        "📊 信号追踪", 
        "📉 信号验证", 
        "📧 历史追踪",
        "🔄 Baseline对比"
    ])
    
    with tab1:
        render_watchlist_tracking_tab()
    
    with tab2:
        render_signal_health_monitor()
    
    with tab3:
        render_signal_tracker_page()
    
    with tab4:
        render_signal_performance_page()
    
    with tab5:
        render_historical_tracking_tab()
    
    with tab6:
        render_baseline_comparison_page()


def render_watchlist_tracking_tab():
    """👁️ 观察列表追踪 - 持续跟踪已发现的机会股票"""
    import plotly.graph_objects as go
    
    st.subheader("👁️ 观察列表追踪")
    st.caption("持续关注已发现机会的股票，实时监控信号变化、卖出点、做T时机")
    
    # 侧边栏设置
    with st.sidebar:
        st.divider()
        st.subheader("👁️ 追踪设置")
        
        market_choice = st.radio(
            "市场", 
            ["🇺🇸 美股", "🇨🇳 A股"], 
            horizontal=True, 
            key="watchlist_market"
        )
        market = "US" if "美股" in market_choice else "CN"
    
    try:
        from services.signal_tracker import (
            get_watchlist, add_to_watchlist, remove_from_watchlist,
            get_signal_history, analyze_sell_signals, analyze_t_trade_opportunity,
            get_unread_alerts, mark_alert_read, get_tracking_summary, record_signal
        )
    except ImportError as e:
        st.error(f"追踪模块导入失败: {e}")
        return
    
    # 获取数据
    watchlist = get_watchlist(market=market)
    tracking_summary = get_tracking_summary(market=market)
    unread_alerts = get_unread_alerts(market=market)
    
    # === 顶部: 追踪概览 ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👁️ 观察中", f"{len(watchlist)} 只")
    
    with col2:
        buy_signals = tracking_summary.get('buy_signals', 0)
        st.metric("🟢 买入信号", f"{buy_signals} 条", 
                  delta="有机会" if buy_signals > 0 else None)
    
    with col3:
        sell_signals = tracking_summary.get('sell_signals', 0)
        st.metric("🔴 卖出信号", f"{sell_signals} 条",
                  delta="需关注" if sell_signals > 0 else None,
                  delta_color="inverse" if sell_signals > 0 else "off")
    
    with col4:
        st.metric("🔔 未读提醒", f"{len(unread_alerts)} 条")
    
    st.divider()
    
    # === 未读提醒 ===
    if unread_alerts:
        with st.expander(f"🔔 未读提醒 ({len(unread_alerts)} 条)", expanded=True):
            for alert in unread_alerts[:10]:
                urgency_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(alert['urgency'], '⚪')
                
                col1, col2, col3 = st.columns([2, 5, 1])
                with col1:
                    st.markdown(f"**{urgency_icon} {alert['symbol']}**")
                with col2:
                    st.markdown(f"{alert['message']}")
                    st.caption(f"{alert['alert_date']} | {alert['alert_type']}")
                with col3:
                    if st.button("✓", key=f"read_{alert['id']}"):
                        mark_alert_read(alert['id'])
                        st.rerun()
    
    # === 添加观察 ===
    with st.expander("➕ 添加股票到观察列表", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_symbol = st.text_input("股票代码", placeholder="NVDA / 600519.SH", key="add_symbol")
            entry_price = st.number_input("入场价", min_value=0.0, step=0.01, key="add_entry")
        
        with col2:
            target_price = st.number_input("目标价 (止盈)", min_value=0.0, step=0.01, key="add_target")
            stop_loss = st.number_input("止损价", min_value=0.0, step=0.01, key="add_stop")
        
        notes = st.text_input("备注", placeholder="买入理由...", key="add_notes")
        
        if st.button("➕ 添加", type="primary"):
            if new_symbol:
                add_to_watchlist(
                    symbol=new_symbol.upper(),
                    market=market,
                    entry_price=entry_price if entry_price > 0 else None,
                    target_price=target_price if target_price > 0 else None,
                    stop_loss=stop_loss if stop_loss > 0 else None,
                    notes=notes
                )
                st.success(f"✅ {new_symbol.upper()} 已添加")
                st.rerun()
            else:
                st.warning("请输入股票代码")
    
    # === 观察列表详情 ===
    if not watchlist:
        st.info("👆 观察列表为空，点击上方添加股票开始追踪")
        return
    
    st.markdown("### 📋 观察列表详情")
    
    # 获取最新扫描数据
    dates = _cached_scanned_dates(market=market)
    latest_date = dates[0] if dates else None
    latest_scan = {}
    
    if latest_date:
        scan_results = _cached_scan_results(scan_date=latest_date, market=market, limit=1000)
        for r in scan_results:
            latest_scan[r['symbol']] = r
    
    # 为每只股票创建追踪卡片
    for item in watchlist:
        symbol = item['symbol']
        entry_price = item.get('entry_price', 0) or 0
        target_price = item.get('target_price', 0) or (entry_price * 1.15 if entry_price else 0)
        stop_loss = item.get('stop_loss', 0) or (entry_price * 0.92 if entry_price else 0)
        added_date = item.get('added_date', '')
        notes = item.get('notes', '')
        
        # 获取最新数据
        scan_data = latest_scan.get(symbol, {})
        current_price = scan_data.get('price', entry_price) or entry_price
        blue_daily = scan_data.get('blue_daily', 0) or 0
        blue_weekly = scan_data.get('blue_weekly', 0) or 0
        heima = scan_data.get('heima', 0) or 0
        volume = scan_data.get('volume', 0) or 0
        
        # 计算盈亏
        pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        # 分析卖出信号
        sell_analysis = analyze_sell_signals(
            symbol, market, current_price, entry_price,
            target_price, stop_loss, blue_daily, blue_weekly
        )
        
        # 卡片样式
        urgency = sell_analysis['sell_urgency']
        border_color = {
            'critical': '#ff4444', 'high': '#ff8800', 
            'medium': '#ffcc00', 'low': '#44ff44', 'none': '#666666'
        }.get(urgency, '#666666')
        
        st.markdown(f"""
        <div style="border-left: 4px solid {border_color}; padding-left: 15px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        
        # 标题行
        col1, col2, col3 = st.columns([3, 5, 2])
        
        with col1:
            urgency_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'none': '⚪'}.get(urgency, '⚪')
            st.markdown(f"### {urgency_icon} {symbol}")
            st.caption(f"加入: {added_date}")
            if notes:
                st.caption(f"📝 {notes}")
        
        with col2:
            # 信号状态
            sub_cols = st.columns(4)
            price_symbol = "¥" if market == "CN" else "$"
            
            with sub_cols[0]:
                pnl_color = "green" if pnl_pct >= 0 else "red"
                st.markdown(f"**现价**")
                st.markdown(f"{price_symbol}{current_price:.2f}")
                st.markdown(f"<span style='color:{pnl_color}'>{pnl_pct:+.1f}%</span>", unsafe_allow_html=True)
            
            with sub_cols[1]:
                blue_color = "green" if blue_daily >= 100 else ("orange" if blue_daily >= 50 else "red")
                st.markdown(f"**日BLUE**")
                st.markdown(f"<span style='color:{blue_color}'>{blue_daily:.0f}</span>", unsafe_allow_html=True)
            
            with sub_cols[2]:
                st.markdown(f"**周BLUE**")
                st.markdown(f"{blue_weekly:.0f}")
            
            with sub_cols[3]:
                st.markdown(f"**黑马**")
                st.markdown("🐴" if heima else "-")
        
        with col3:
            # 交易计划
            st.markdown(f"🎯 {price_symbol}{target_price:.2f}")
            st.markdown(f"🛑 {price_symbol}{stop_loss:.2f}")
        
        # 卖出建议
        if sell_analysis['should_sell'] or sell_analysis['reasons']:
            with st.container():
                action_text = sell_analysis.get('recommended_action', 'hold')
                action_display = {
                    'sell_now': '🔴 建议立即卖出',
                    'take_profit': '🟢 已达止盈目标',
                    'consider_sell': '🟡 考虑卖出',
                    'consider_partial_sell': '🟡 考虑部分卖出',
                    'hold': '✅ 继续持有'
                }.get(action_text, '⚪ ' + action_text)
                
                st.markdown(f"**{action_display}**")
                
                for reason in sell_analysis['reasons']:
                    st.markdown(f"  • {reason}")
        
        # 操作按钮
        btn_cols = st.columns([1, 1, 1, 3])
        
        with btn_cols[0]:
            if st.button("📊 详情", key=f"detail_{symbol}"):
                st.session_state['stock_symbol'] = symbol
                st.info(f"请前往「个股查询」查看 {symbol} 详情")
        
        with btn_cols[1]:
            if st.button("💰 模拟买", key=f"sim_buy_{symbol}"):
                st.info("请前往「组合管理」执行模拟交易")
        
        with btn_cols[2]:
            if st.button("❌ 移除", key=f"del_{symbol}"):
                remove_from_watchlist(symbol, market)
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
    
    # === 信号历史图表 ===
    st.markdown("### 📈 信号历史对比")
    
    if len(watchlist) > 0:
        selected_symbol = st.selectbox(
            "选择股票查看历史",
            [w['symbol'] for w in watchlist],
            key="history_select"
        )
        
        if selected_symbol:
            history = get_signal_history(selected_symbol, market, days=30)
            
            if history:
                hist_df = pd.DataFrame(history)
                hist_df['record_date'] = pd.to_datetime(hist_df['record_date'])
                hist_df = hist_df.sort_values('record_date')
                
                # 创建图表
                fig = go.Figure()
                
                # 价格线
                if 'price' in hist_df.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_df['record_date'],
                        y=hist_df['price'],
                        name='价格',
                        line=dict(color='white', width=2),
                        yaxis='y'
                    ))
                
                # BLUE 指标
                if 'blue_daily' in hist_df.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_df['record_date'],
                        y=hist_df['blue_daily'],
                        name='日BLUE',
                        line=dict(color='#00ff88', width=1.5),
                        yaxis='y2'
                    ))
                
                if 'blue_weekly' in hist_df.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_df['record_date'],
                        y=hist_df['blue_weekly'],
                        name='周BLUE',
                        line=dict(color='#ffaa00', width=1.5),
                        yaxis='y2'
                    ))
                
                # 买入线 (BLUE=100)
                fig.add_hline(y=100, line_dash="dash", line_color="green", 
                              annotation_text="BLUE买入线", yref='y2')
                
                fig.update_layout(
                    title=f"{selected_symbol} 信号历史 (30日)",
                    xaxis_title="日期",
                    yaxis=dict(title="价格", side='left'),
                    yaxis2=dict(title="BLUE", overlaying='y', side='right'),
                    template='plotly_dark',
                    height=400,
                    legend=dict(x=0, y=1.1, orientation='h')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"暂无 {selected_symbol} 的历史数据")


def render_historical_tracking_tab():
    """📧 历史信号追踪 - 类似邮件报告的内容"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.subheader("📧 历史信号追踪报告")
    st.caption("追踪过去30天每个信号的后续表现")
    
    # 侧边栏设置
    with st.sidebar:
        st.subheader("📧 追踪设置")
        
        market_choice = st.radio(
            "市场", 
            ["🇺🇸 美股", "🇨🇳 A股"], 
            horizontal=True, 
            key="hist_track_market"
        )
        market = "US" if "美股" in market_choice else "CN"
        
        days = st.slider("追踪天数", 7, 60, 30, key="hist_track_days")
        min_blue = st.slider("最低 BLUE 阈值", 100, 180, 130, key="hist_track_blue")
        
        generate_btn = st.button("📊 生成报告", type="primary", use_container_width=True)
    
    if not generate_btn:
        st.info("👈 设置参数后点击「生成报告」查看历史信号表现")
        
        st.markdown("""
        ### 📋 报告内容
        
        - **信号统计**: 总数、胜率、平均收益
        - **各周期表现**: D+1, D+3, D+5, D+10 收益
        - **最佳/最差信号**: 表现最好和最差的股票
        - **详细列表**: 每个信号的完整表现
        
        ### 💡 说明
        
        - BLUE ≥ 130 被视为有效信号
        - 胜率 = 当前盈利的信号占比
        - 各周期收益 = 信号后第N个交易日的累计收益
        """)
        return
    
    # 获取数据
    from db.database import query_scan_results, get_scanned_dates
    from data_fetcher import get_stock_data
    
    with st.spinner("获取历史信号..."):
        dates = _cached_scanned_dates(market=market)
        if not dates:
            st.error("没有找到扫描数据")
            return
        
        all_signals = []
        for date in dates[:days]:
            results = query_scan_results(scan_date=date, market=market, limit=100)
            for r in results:
                blue = r.get('blue_daily', 0) or 0
                if blue >= min_blue:
                    all_signals.append({
                        'symbol': r['symbol'],
                        'signal_date': date,
                        'signal_price': r.get('price', 0),
                        'blue': blue,
                        'adx': r.get('adx', 0) or 0,
                        'is_heima': r.get('is_heima', False),
                        'is_juedi': r.get('is_juedi', False),
                        'company_name': r.get('company_name', '') or ''
                    })
        
        st.success(f"找到 {len(all_signals)} 个信号")
    
    if not all_signals:
        st.warning("没有找到符合条件的信号")
        return
    
    # 计算收益
    with st.spinner("计算信号收益 (可能需要几分钟)..."):
        results = []
        symbol_cache = {}
        progress = st.progress(0)
        
        for i, sig in enumerate(all_signals[:100]):  # 限制100个避免太慢
            symbol = sig['symbol']
            signal_price = sig['signal_price']
            
            if not signal_price or signal_price <= 0:
                continue
            
            # 获取价格
            if symbol not in symbol_cache:
                try:
                    df = get_stock_data(symbol, market=market, days=60)
                    symbol_cache[symbol] = df
                except:
                    symbol_cache[symbol] = None
            
            df = symbol_cache[symbol]
            if df is None or len(df) < 5:
                continue
            
            try:
                sig_dt = pd.to_datetime(sig['signal_date'])
                future_df = df[df.index > sig_dt]
                
                d1 = (future_df.iloc[0]['Close'] / signal_price - 1) * 100 if len(future_df) >= 1 else None
                d3 = (future_df.iloc[2]['Close'] / signal_price - 1) * 100 if len(future_df) >= 3 else None
                d5 = (future_df.iloc[4]['Close'] / signal_price - 1) * 100 if len(future_df) >= 5 else None
                d10 = (future_df.iloc[9]['Close'] / signal_price - 1) * 100 if len(future_df) >= 10 else None
                
                current_price = df.iloc[-1]['Close']
                current_return = (current_price / signal_price - 1) * 100
                
                results.append({
                    **sig,
                    'current_price': current_price,
                    'current_return': current_return,
                    'D1': d1, 'D3': d3, 'D5': d5, 'D10': d10,
                    'is_winner': current_return > 0
                })
            except:
                pass
            
            progress.progress((i + 1) / min(len(all_signals), 100))
        
        progress.empty()
    
    if not results:
        st.error("无法计算信号收益")
        return
    
    df = pd.DataFrame(results)
    
    # === 统计卡片 ===
    st.markdown("### 📊 整体统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总信号数", len(df))
    
    with col2:
        winners = len(df[df['is_winner'] == True])
        win_rate = winners / len(df) * 100
        st.metric("胜率", f"{win_rate:.1f}%", delta="盈利多" if win_rate > 50 else "亏损多")
    
    with col3:
        avg_return = df['current_return'].mean()
        st.metric("平均收益", f"{avg_return:+.2f}%")
    
    with col4:
        st.metric("盈/亏", f"{winners}/{len(df) - winners}")
    
    st.divider()
    
    # === 各周期收益 ===
    st.markdown("### 📈 各周期平均收益")
    
    col1, col2, col3, col4 = st.columns(4)
    
    d1_avg = df['D1'].dropna().mean() if len(df['D1'].dropna()) > 0 else 0
    d3_avg = df['D3'].dropna().mean() if len(df['D3'].dropna()) > 0 else 0
    d5_avg = df['D5'].dropna().mean() if len(df['D5'].dropna()) > 0 else 0
    d10_avg = df['D10'].dropna().mean() if len(df['D10'].dropna()) > 0 else 0
    
    with col1:
        st.metric("D+1", f"{d1_avg:+.2f}%")
    with col2:
        st.metric("D+3", f"{d3_avg:+.2f}%")
    with col3:
        st.metric("D+5", f"{d5_avg:+.2f}%")
    with col4:
        st.metric("D+10", f"{d10_avg:+.2f}%")
    
    # 收益曲线图
    returns_data = pd.DataFrame({
        '周期': ['D+1', 'D+3', 'D+5', 'D+10'],
        '平均收益': [d1_avg, d3_avg, d5_avg, d10_avg]
    })
    
    fig = px.bar(returns_data, x='周期', y='平均收益', 
                 color='平均收益',
                 color_continuous_scale=['red', 'gray', 'green'],
                 title="各周期平均收益")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # === 最佳/最差 ===
    st.markdown("### 🏆 最佳 vs 最差")
    
    best = df.loc[df['current_return'].idxmax()]
    worst = df.loc[df['current_return'].idxmin()]
    price_sym = "$" if market == "US" else "¥"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🥇 **{best['symbol']}** {best.get('company_name', '')[:15]}")
        st.write(f"信号日期: {best['signal_date']}")
        st.write(f"信号价: {price_sym}{best['signal_price']:.2f}")
        st.write(f"当前价: {price_sym}{best['current_price']:.2f}")
        st.write(f"**收益: +{best['current_return']:.1f}%**")
    
    with col2:
        st.error(f"❌ **{worst['symbol']}** {worst.get('company_name', '')[:15]}")
        st.write(f"信号日期: {worst['signal_date']}")
        st.write(f"信号价: {price_sym}{worst['signal_price']:.2f}")
        st.write(f"当前价: {price_sym}{worst['current_price']:.2f}")
        st.write(f"**收益: {worst['current_return']:.1f}%**")
    
    st.divider()
    
    # === 详细列表 ===
    st.markdown("### 📋 信号详情")
    
    display_df = df[['signal_date', 'symbol', 'company_name', 'blue', 'signal_price', 
                     'D1', 'D3', 'D5', 'D10', 'current_return']].copy()
    display_df.columns = ['日期', '代码', '名称', 'BLUE', '信号价', 'D+1', 'D+3', 'D+5', 'D+10', '当前收益']
    
    # 格式化
    for col in ['D+1', 'D+3', 'D+5', 'D+10', '当前收益']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
    
    display_df['信号价'] = display_df['信号价'].apply(lambda x: f"{price_sym}{x:.2f}")
    display_df['名称'] = display_df['名称'].apply(lambda x: x[:12] if x else '')
    
    st.dataframe(display_df.sort_values('日期', ascending=False), 
                 hide_index=True, 
                 use_container_width=True,
                 height=400)


def render_signal_health_monitor():
    """🩺 信号健康度监控"""
    import plotly.graph_objects as go
    import plotly.express as px
    
    st.subheader("🩺 信号衰减监控")
    st.caption("实时追踪各类信号的胜率变化，及时发现信号失效")
    
    # 参数设置
    col1, col2, col3 = st.columns(3)
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="health_market")
    with col2:
        min_blue = st.slider("BLUE 阈值", 50, 150, 100, key="health_blue")
    with col3:
        holding_days = st.selectbox("持有天数", [3, 5, 10, 20], index=1, key="health_days")
    
    # 获取健康度数据
    try:
        from services.signal_monitor import SignalMonitor, SignalType, HealthStatus
        
        with st.spinner("正在分析信号健康度..."):
            monitor = SignalMonitor(market=market, holding_days=holding_days)
            all_health = monitor.get_all_signals_health(min_blue=min_blue)
        
        # === 整体状态卡片 ===
        st.markdown("### 📊 整体状态")
        
        status_counts = {'healthy': 0, 'warning': 0, 'critical': 0, 'unknown': 0}
        for health in all_health.values():
            status_counts[health.status.value] += 1
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("🟢 健康", status_counts['healthy'])
        with cols[1]:
            st.metric("🟡 关注", status_counts['warning'])
        with cols[2]:
            st.metric("🔴 衰减", status_counts['critical'])
        with cols[3]:
            st.metric("⚪ 未知", status_counts['unknown'])
        
        st.divider()
        
        # === 各信号详情 ===
        st.markdown("### 📋 各信号健康度")
        
        signal_names = {
            SignalType.DAILY_BLUE: "日 BLUE",
            SignalType.WEEKLY_BLUE: "周 BLUE",
            SignalType.MONTHLY_BLUE: "月 BLUE",
            SignalType.DAILY_WEEKLY: "日+周共振",
            SignalType.HEIMA: "黑马信号",
            SignalType.ALL_RESONANCE: "全共振"
        }
        
        status_icons = {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.WARNING: "🟡",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.UNKNOWN: "⚪"
        }
        
        for signal_type, health in all_health.items():
            with st.expander(f"{status_icons[health.status]} {signal_names[signal_type]} - {health.status.value.upper()}", expanded=health.status != HealthStatus.UNKNOWN):
                
                if health.status == HealthStatus.UNKNOWN:
                    st.info("数据不足，无法评估")
                    continue
                
                # 胜率指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("7天胜率", f"{health.win_rate_7d:.0%}", 
                             delta=f"{(health.win_rate_7d - health.win_rate_90d)*100:+.0f}pp" if health.win_rate_90d > 0 else None)
                with col2:
                    st.metric("30天胜率", f"{health.win_rate_30d:.0%}",
                             delta=f"{(health.win_rate_30d - health.win_rate_90d)*100:+.0f}pp" if health.win_rate_90d > 0 else None)
                with col3:
                    st.metric("90天胜率", f"{health.win_rate_90d:.0%}")
                with col4:
                    decay_color = "normal" if health.decay_ratio >= 0.9 else "inverse"
                    st.metric("衰减比率", f"{health.decay_ratio:.0%}", delta_color=decay_color)
                
                # 收益指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("7天平均收益", f"{health.avg_return_7d:.1f}%")
                with col2:
                    st.metric("30天平均收益", f"{health.avg_return_30d:.1f}%")
                with col3:
                    st.metric("总平均收益", f"{health.avg_return_all:.1f}%")
                with col4:
                    st.metric("样本量(30天)", f"{health.sample_30d}")
                
                # 趋势图标
                trend_icons = {"improving": "📈 改善", "stable": "➡️ 稳定", "declining": "📉 下降"}
                st.caption(f"趋势: {trend_icons.get(health.trend, health.trend)}")
                
                # 建议
                if health.status == HealthStatus.CRITICAL:
                    st.error(f"💡 建议: {health.recommendation}")
                elif health.status == HealthStatus.WARNING:
                    st.warning(f"💡 建议: {health.recommendation}")
                else:
                    st.success(f"💡 建议: {health.recommendation}")
        
        st.divider()
        
        # === 胜率对比图 ===
        st.markdown("### 📈 胜率对比")
        
        # 准备数据
        chart_data = []
        for signal_type, health in all_health.items():
            if health.status != HealthStatus.UNKNOWN:
                chart_data.append({
                    '信号类型': signal_names[signal_type],
                    '7天': health.win_rate_7d * 100,
                    '30天': health.win_rate_30d * 100,
                    '90天': health.win_rate_90d * 100
                })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='7天', x=chart_df['信号类型'], y=chart_df['7天'], marker_color='#636EFA'))
            fig.add_trace(go.Bar(name='30天', x=chart_df['信号类型'], y=chart_df['30天'], marker_color='#EF553B'))
            fig.add_trace(go.Bar(name='90天', x=chart_df['信号类型'], y=chart_df['90天'], marker_color='#00CC96'))
            
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% 基准线")
            
            fig.update_layout(
                title="各信号胜率对比 (%)",
                barmode='group',
                yaxis_title="胜率 %",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # === 告警汇总 ===
        alerts = monitor.get_decay_alerts(min_blue)
        if alerts:
            st.markdown("### ⚠️ 告警")
            for alert in alerts:
                if alert.status == HealthStatus.CRITICAL:
                    st.error(f"🔴 **{signal_names[alert.signal_type]}**: {alert.recommendation}")
                else:
                    st.warning(f"🟡 **{signal_names[alert.signal_type]}**: {alert.recommendation}")
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_portfolio_management_page():
    """💼 组合管理 - 合并: 持仓管理 + 风控仪表盘 + 模拟交易 + 观察追踪"""
    st.header("💼 组合管理")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🛡️ 风控仪表盘", "💰 持仓管理", "🎮 模拟交易", "👁️ 观察追踪"])
    
    with tab1:
        render_risk_dashboard()
    
    with tab2:
        render_portfolio_tab()
    
    with tab3:
        render_paper_trading_tab()
    
    with tab4:
        render_watchlist_tracking_tab()


def render_risk_dashboard():
    """🛡️ 风控仪表盘 - 基于真实持仓数据"""
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, timedelta
    import numpy as np
    
    st.subheader("🛡️ 风险控制中心")
    
    # === 数据源选择 ===
    st.markdown("#### 📂 选择分析对象")
    
    source_options = {
        "🎮 模拟持仓": "paper",
        "💰 实盘持仓": "real",
        "📊 每日机会 (全部)": "daily_all",
        "🔵 仅日BLUE信号": "daily_blue",
        "🔷 日+周BLUE共振": "daily_weekly",
        "🔶 月BLUE信号": "monthly_blue",
        "🐴 黑马信号": "heima",
        "⭐ 全条件共振 (日+周+月+黑马)": "all_resonance"
    }
    
    data_source = st.selectbox(
        "数据来源",
        list(source_options.keys()),
        help="选择要分析的持仓/信号数据"
    )
    
    source_key = source_options[data_source]
    
    # === 信号筛选参数 ===
    if source_key not in ['paper', 'real']:
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            market_filter = st.selectbox("市场", ["US", "CN"], key="risk_market")
        with filter_col2:
            days_back = st.slider("回看天数", 1, 30, 7, key="risk_days")
        with filter_col3:
            min_blue = st.slider("最低BLUE", 50, 150, 100, key="risk_blue")
    
    # === 获取数据 ===
    holdings = {}
    positions = []
    total_value = 0
    
    try:
        if source_key == "paper":
            # 模拟持仓
            from services.portfolio_service import get_paper_account
            account = get_paper_account()
            if account and account.get('positions'):
                positions = account['positions']
                total_value = account.get('total_equity', 0)
                
        elif source_key == "real":
            # 实盘持仓
            from db.database import get_portfolio
            from services.portfolio_service import get_current_price
            portfolio = get_portfolio()
            if portfolio:
                for p in portfolio:
                    price = get_current_price(p['symbol'], p.get('market', 'US'))
                    if price:
                        p['market_value'] = price * p['shares']
                        p['current_price'] = price
                    else:
                        p['market_value'] = p.get('cost_basis', 0) * p['shares']
                    total_value += p['market_value']
                positions = portfolio
                
        else:
            # 从扫描信号获取
            from db.database import query_scan_results
            from services.portfolio_service import get_current_price
            from datetime import date, timedelta
            
            # 获取最近 N 天的扫描结果
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            all_signals = []
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                try:
                    results = query_scan_results(date_str, market=market_filter, min_blue=min_blue)
                    if results:
                        for r in results:
                            r['scan_date'] = date_str
                        all_signals.extend(results)
                except:
                    pass
                current_date += timedelta(days=1)
            
            if all_signals:
                # 根据策略筛选
                filtered_signals = []
                
                for sig in all_signals:
                    # 字段名兼容 (数据库用小写，CSV用大写)
                    day_blue = sig.get('blue_daily', sig.get('Day_BLUE', 0)) or 0
                    week_blue = sig.get('blue_weekly', sig.get('Week_BLUE', 0)) or 0
                    month_blue = sig.get('blue_monthly', sig.get('Month_BLUE', 0)) or 0
                    heima = sig.get('is_heima', sig.get('Heima', False))
                    
                    if source_key == "daily_all":
                        # 所有信号
                        if day_blue >= min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "daily_blue":
                        # 仅日BLUE
                        if day_blue >= min_blue and week_blue < min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "daily_weekly":
                        # 日+周共振
                        if day_blue >= min_blue and week_blue >= min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "monthly_blue":
                        # 月BLUE
                        if month_blue >= min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "heima":
                        # 黑马信号
                        if heima:
                            filtered_signals.append(sig)
                            
                    elif source_key == "all_resonance":
                        # 全条件共振
                        if day_blue >= min_blue and week_blue >= min_blue and (month_blue >= min_blue or heima):
                            filtered_signals.append(sig)
                
                # 去重 (同一只股票只保留最新，按 BLUE 值排序)
                symbol_latest = {}
                for sig in filtered_signals:
                    sym = sig.get('symbol', sig.get('Symbol', ''))
                    if sym:
                        if sym not in symbol_latest or sig['scan_date'] > symbol_latest[sym]['scan_date']:
                            symbol_latest[sym] = sig
                
                # 按 blue_daily 排序，取 Top N
                MAX_POSITIONS = 20  # 限制最多分析 20 只
                sorted_symbols = sorted(
                    symbol_latest.items(),
                    key=lambda x: x[1].get('blue_daily', x[1].get('Day_BLUE', 0)) or 0,
                    reverse=True
                )[:MAX_POSITIONS]
                
                st.info(f"📊 筛选: {len(all_signals)} 条信号 → {len(filtered_signals)} 符合 → {len(symbol_latest)} 只股票 → Top {len(sorted_symbols)} (按BLUE排序)")
                
                # 转换为持仓格式 (等权重)
                if sorted_symbols:
                    equal_value = 100000 / len(sorted_symbols)  # 10万等分
                    
                    progress_bar = st.progress(0, text="正在获取价格数据...")
                    
                    for i, (sym, sig) in enumerate(sorted_symbols):
                        progress_bar.progress((i + 1) / len(sorted_symbols), text=f"获取 {sym} 价格...")
                        
                        # 先尝试用扫描时的价格
                        price = sig.get('price', sig.get('Close', None))
                        if not price:
                            price = get_current_price(sym, market_filter)
                        
                        if price and price > 0:
                            shares = int(equal_value / price)
                            market_value = shares * price
                            
                            positions.append({
                                'symbol': sym,
                                'shares': shares,
                                'avg_cost': price,
                                'current_price': price,
                                'market_value': market_value,
                                'market': market_filter,
                                'day_blue': sig.get('blue_daily', sig.get('Day_BLUE', 0)),
                                'week_blue': sig.get('blue_weekly', sig.get('Week_BLUE', 0)),
                                'unrealized_pnl_pct': 0
                            })
                            total_value += market_value
                    
                    progress_bar.empty()
                
    except Exception as e:
        st.warning(f"获取数据失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # 检查是否有持仓
    if not positions:
        st.info("📭 暂无持仓数据")
        st.markdown("""
        请先在以下位置添加持仓:
        - **模拟交易** Tab: 使用虚拟资金买入股票
        - **持仓管理** Tab: 手动添加实盘持仓
        """)
        
        # 显示仓位计算器作为替代
        st.divider()
        render_position_calculator()
        return
    
    # 计算持仓权重
    for pos in positions:
        symbol = pos.get('symbol', 'Unknown')
        market_value = pos.get('market_value', 0)
        if total_value > 0:
            holdings[symbol] = market_value / total_value
    
    symbols = list(holdings.keys())
    
    st.success(f"✅ 已加载 {len(positions)} 个持仓，总市值 ${total_value:,.0f}")
    
    # === 获取历史数据计算风险指标 ===
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_returns_from_scan_history(symbols_tuple, market, days_back=60):
        """从扫描历史数据计算收益率 (不调用外部 API)"""
        from db.database import get_connection
        from datetime import date, timedelta
        
        returns_dict = {}
        symbols_list = list(symbols_tuple)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # 获取每只股票的历史扫描价格
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        for sym in symbols_list:
            try:
                cursor.execute("""
                    SELECT scan_date, price 
                    FROM scan_results 
                    WHERE symbol = ? AND market = ? AND scan_date >= ? 
                    ORDER BY scan_date
                """, (sym, market, start_date.strftime('%Y-%m-%d')))
                
                rows = cursor.fetchall()
                if len(rows) >= 2:
                    prices = pd.Series(
                        {row['scan_date']: row['price'] for row in rows if row['price']}
                    )
                    if len(prices) >= 2:
                        returns = prices.pct_change().dropna()
                        if len(returns) >= 1:
                            returns_dict[sym] = returns
            except:
                continue
        
        conn.close()
        return returns_dict
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_returns_from_api(symbols_tuple, market, days=60):
        """从 API 获取收益率数据 (备选)"""
        from data_fetcher import get_us_stock_data, get_cn_stock_data
        import time
        
        returns_dict = {}
        symbols_list = list(symbols_tuple)
        
        for i, sym in enumerate(symbols_list[:10]):  # 最多取 10 只，避免 rate limit
            try:
                if market == 'CN' or sym.endswith('.SH') or sym.endswith('.SZ'):
                    df = get_cn_stock_data(sym, days=days)
                else:
                    df = get_us_stock_data(sym, days=days)
                
                if df is not None and len(df) > 10:
                    returns_dict[sym] = df['Close'].pct_change().dropna()
                
                if i < len(symbols_list) - 1:
                    time.sleep(0.2)
            except:
                continue
        
        return returns_dict
    
    # 获取风险数据
    current_market = market_filter if source_key not in ['paper', 'real'] else 'US'
    
    # 先尝试扫描历史
    with st.spinner("正在计算风险指标..."):
        returns_data = get_returns_from_scan_history(tuple(symbols), current_market, days_back=60)
    
    # 如果扫描历史不足，尝试 API (只取前 5 只)
    if len(returns_data) < 2:
        st.caption("📊 扫描历史稀疏，从 API 获取主要持仓数据...")
        returns_data = get_returns_from_api(tuple(symbols[:5]), current_market, days=60)
    
    st.caption(f"📈 获取到 {len(returns_data)} 只股票的历史数据")
    
    # === 第一行: 核心风险指标 ===
    st.markdown("### 📊 组合风险概览")
    
    # 计算组合收益率
    if returns_data and len(returns_data) > 0:
        # 对齐日期
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) > 20:
            # 计算组合加权收益
            weight_array = np.array([holdings.get(s, 0) for s in returns_df.columns])
            weight_array = weight_array / weight_array.sum()  # 归一化
            
            portfolio_returns = (returns_df * weight_array).sum(axis=1)
            
            # 计算风险指标
            var_95 = np.percentile(portfolio_returns, 5) * 100
            
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            
            excess_returns = portfolio_returns - 0.02/252  # 假设无风险利率 2%
            sharpe = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
        else:
            var_95, max_dd, volatility, sharpe = -2.0, -5.0, 20.0, 1.0
            st.warning("历史数据不足，使用估算值")
    else:
        var_95, max_dd, volatility, sharpe = -2.0, -5.0, 20.0, 1.0
        st.warning("无法获取历史数据，使用估算值")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "VaR (95%, 1天)",
            f"{var_95:.2f}%",
            delta="正常" if var_95 > -5 else "警告",
            delta_color="normal" if var_95 > -5 else "inverse"
        )
        st.caption("单日最大损失估计")
    
    with col2:
        st.metric(
            "最大回撤",
            f"{max_dd:.1f}%",
            delta="可控" if max_dd > -15 else "需关注",
            delta_color="normal" if max_dd > -15 else "inverse"
        )
    
    with col3:
        st.metric(
            "年化波动率",
            f"{volatility:.1f}%",
            delta="中等" if volatility < 25 else "偏高",
            delta_color="normal" if volatility < 25 else "inverse"
        )
    
    with col4:
        st.metric(
            "Sharpe 比率",
            f"{sharpe:.2f}",
            delta="优秀" if sharpe > 1.5 else ("一般" if sharpe > 0.5 else "差"),
            delta_color="normal" if sharpe > 1.0 else "inverse"
        )
    
    st.divider()
    
    # === 第二行: 持仓集中度 + 持仓明细 ===
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📈 持仓集中度")
        
        # 饼图 - 使用真实持仓
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(holdings.keys()),
            values=[v * 100 for v in holdings.values()],
            hole=0.4,
            textinfo='label+percent',
            marker_colors=px.colors.qualitative.Set3
        )])
        fig_pie.update_layout(
            title=f"持仓分布 (共 {len(holdings)} 只)",
            height=300,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 集中度警告
        if holdings:
            max_symbol = max(holdings, key=holdings.get)
            max_weight = holdings[max_symbol]
            
            if max_weight > 0.25:
                st.error(f"🔴 单股集中度过高: {max_symbol} = {max_weight:.0%} (建议 < 25%)")
            elif max_weight > 0.20:
                st.warning(f"⚠️ 单股集中度偏高: {max_symbol} = {max_weight:.0%}")
            else:
                st.success(f"✅ 集中度正常: 最大持仓 {max_symbol} = {max_weight:.0%}")
    
    with col_right:
        st.markdown("### 📋 持仓明细")
        
        # 持仓表格
        pos_df = pd.DataFrame([{
            '代码': p.get('symbol'),
            '股数': p.get('shares'),
            '市值': f"${p.get('market_value', 0):,.0f}",
            '权重': f"{holdings.get(p.get('symbol'), 0):.1%}",
            '盈亏': f"{p.get('unrealized_pnl_pct', 0):.1f}%"
        } for p in positions])
        
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # === 第三行: 相关性矩阵 + 回撤曲线 ===
    col_corr, col_dd = st.columns(2)
    
    with col_corr:
        st.markdown("### 🔗 持仓相关性")
        
        if returns_data and len(returns_data) >= 2:
            # 使用 pairwise 相关性 (允许不同日期)
            returns_df = pd.DataFrame(returns_data)
            
            # 计算 pairwise 相关性 (使用重叠日期)
            corr_matrix = returns_df.corr(min_periods=2)  # 至少 2 个重叠点
            
            # 检查是否有有效的相关性
            valid_corr = corr_matrix.dropna(how='all').dropna(axis=1, how='all')
            
            if len(valid_corr) >= 2:
                fig_corr = px.imshow(
                    valid_corr.values,
                    x=valid_corr.columns.tolist(),
                    y=valid_corr.index.tolist(),
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    title=f"相关性矩阵 ({len(valid_corr)} 只股票)",
                    zmin=-1, zmax=1
                )
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # 高相关性警告
                high_corr_pairs = []
                cols = valid_corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        val = valid_corr.iloc[i, j]
                        if pd.notna(val) and val > 0.75:
                            high_corr_pairs.append((cols[i], cols[j], val))
                
                if high_corr_pairs:
                    st.warning(f"⚠️ 高相关性: {', '.join([f'{p[0]}-{p[1]}({p[2]:.2f})' for p in high_corr_pairs[:3]])}")
                else:
                    st.success("✅ 持仓分散度良好")
            else:
                st.info("📊 数据重叠不足，显示持仓列表")
                st.dataframe(pd.DataFrame({
                    '股票': list(returns_data.keys()),
                    '数据点': [len(v) for v in returns_data.values()]
                }), hide_index=True)
        else:
            st.info("需要至少 2 个持仓才能计算相关性")
    
    with col_dd:
        st.markdown("### 📉 个股收益分布")
        
        if returns_data and len(returns_data) > 0:
            # 计算每只股票的总收益和统计
            stock_stats = []
            for sym, rets in returns_data.items():
                if len(rets) > 0:
                    total_ret = (1 + rets).prod() - 1
                    avg_ret = rets.mean()
                    volatility = rets.std()
                    stock_stats.append({
                        'symbol': sym,
                        'total_return': total_ret * 100,
                        'avg_daily': avg_ret * 100,
                        'volatility': volatility * 100,
                        'days': len(rets)
                    })
            
            if stock_stats:
                stats_df = pd.DataFrame(stock_stats)
                
                # 收益分布柱状图
                fig_returns = go.Figure()
                colors = ['green' if r >= 0 else 'red' for r in stats_df['total_return']]
                fig_returns.add_trace(go.Bar(
                    x=stats_df['symbol'],
                    y=stats_df['total_return'],
                    marker_color=colors,
                    text=[f"{r:.1f}%" for r in stats_df['total_return']],
                    textposition='outside'
                ))
                fig_returns.add_hline(y=0, line_color="gray")
                fig_returns.update_layout(
                    title="各股票累计收益 (%)",
                    xaxis_title="股票",
                    yaxis_title="收益 %",
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # 统计摘要
                avg_return = stats_df['total_return'].mean()
                win_count = (stats_df['total_return'] > 0).sum()
                win_rate = win_count / len(stats_df) * 100
                
                st.caption(f"📊 平均收益: {avg_return:.1f}% | 胜率: {win_rate:.0f}% ({win_count}/{len(stats_df)})")
            else:
                st.info("数据不足")
        else:
            st.info("无历史数据")
    
    st.divider()
    
    # === 仓位计算器 ===
    render_position_calculator()


def render_position_calculator():
    """仓位计算器组件"""
    st.markdown("### 🧮 仓位计算器")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        st.markdown("#### 固定比例法")
        with st.form("position_calc"):
            total_capital = st.number_input("总资金 ($)", value=100000, step=10000)
            risk_per_trade = st.slider("每笔风险比例 (%)", 1, 5, 2) / 100
            entry_price = st.number_input("入场价格", value=150.0, step=1.0)
            stop_loss = st.number_input("止损价格", value=142.0, step=1.0)
            
            if st.form_submit_button("计算仓位"):
                risk_amount = total_capital * risk_per_trade
                risk_per_share = abs(entry_price - stop_loss)
                
                if risk_per_share > 0:
                    shares = int(risk_amount / risk_per_share)
                    position_value = shares * entry_price
                    position_pct = position_value / total_capital
                    
                    st.success(f"""
                    **建议仓位:**
                    - 股数: **{shares:,}** 股
                    - 仓位金额: **${position_value:,.0f}**
                    - 仓位比例: **{position_pct:.1%}**
                    - 最大亏损: **${risk_amount:,.0f}** ({risk_per_trade:.1%})
                    """)
                    
                    if position_pct > 0.20:
                        st.warning("⚠️ 仓位超过 20%，建议分批建仓")
                else:
                    st.error("止损价格不能等于入场价格")
    
    with calc_col2:
        st.markdown("#### 凯利公式")
        with st.form("kelly_calc"):
            win_rate = st.slider("胜率 (%)", 30, 80, 55) / 100
            avg_win = st.number_input("平均盈利 (%)", value=8.0, step=1.0) / 100
            avg_loss = st.number_input("平均亏损 (%)", value=4.0, step=1.0) / 100
            kelly_fraction = st.slider("凯利系数 (保守)", 0.25, 1.0, 0.5, step=0.25)
            
            if st.form_submit_button("计算最优仓位"):
                if avg_loss > 0:
                    # 凯利公式: f = (bp - q) / b
                    b = avg_win / avg_loss  # 赔率
                    p = win_rate
                    q = 1 - p
                    
                    full_kelly = (b * p - q) / b
                    adjusted_kelly = max(0, full_kelly * kelly_fraction)
                    
                    st.success(f"""
                    **凯利公式结果:**
                    - 赔率 (盈亏比): **{b:.2f}**
                    - 完整凯利: **{full_kelly:.1%}**
                    - {kelly_fraction:.0%} 凯利: **{adjusted_kelly:.1%}**
                    
                    建议仓位: **{min(adjusted_kelly, 0.20):.1%}** (上限 20%)
                    """)
                    
                    if full_kelly < 0:
                        st.error("❌ 期望值为负，不建议交易")
                else:
                    st.error("平均亏损必须大于 0")


def render_portfolio_tab():
    """💰 持仓管理 Tab"""
    st.subheader("💰 持仓管理")
    
    # 复用原有的 portfolio 渲染逻辑
    try:
        # 获取持仓数据
        from services.portfolio_service import (
            get_portfolio_summary, 
            get_current_price,
            get_paper_account
        )
        from db.database import get_portfolio, get_trades
        
        portfolio = get_portfolio()
        
        if portfolio:
            summary = get_portfolio_summary()
            
            # 显示汇总
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总成本", f"${summary['total_cost']:,.0f}")
            with col2:
                st.metric("市值", f"${summary['total_value']:,.0f}")
            with col3:
                pnl = summary['unrealized_pnl']
                st.metric("未实现盈亏", f"${pnl:,.0f}", 
                         delta=f"{summary['unrealized_pnl_pct']:.1f}%",
                         delta_color="normal" if pnl >= 0 else "inverse")
            with col4:
                st.metric("持仓数", f"{summary['position_count']}")
            
            # 持仓列表
            st.dataframe(
                pd.DataFrame(portfolio),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("暂无持仓记录")
            
    except Exception as e:
        st.warning(f"持仓数据加载失败: {e}")
        st.info("请先在数据库中添加持仓记录")


def render_paper_trading_tab():
    """🎮 模拟交易 Tab"""
    st.subheader("🎮 模拟交易")
    
    try:
        from services.portfolio_service import (
            get_paper_account,
            paper_buy,
            paper_sell,
            get_paper_trades,
            reset_paper_account,
            get_paper_equity_curve,
            get_paper_monthly_returns,
            list_paper_accounts,
            create_paper_account,
            get_paper_account_config,
            update_paper_account_config,
            get_all_paper_accounts_performance,
            get_paper_account_performance
        )

        sub_accounts = list_paper_accounts()
        sub_names = [a['account_name'] for a in sub_accounts] if sub_accounts else ['default']
        default_account = _get_global_paper_account_name()
        default_index = sub_names.index(default_account) if default_account in sub_names else 0
        selected_account = st.selectbox(
            "策略子账户",
            sub_names,
            index=default_index,
            key="strategy_lab_paper_account_name"
        )
        _set_global_paper_account_name(selected_account)
        account_cfg = get_paper_account_config(selected_account)
        st.caption(
            f"当前风控: 单票≤{float(account_cfg.get('max_single_position_pct', 0.30))*100:.1f}% | "
            f"最大回撤≤{float(account_cfg.get('max_drawdown_pct', 0.20))*100:.1f}%"
        )

        with st.expander("➕ 新建策略子账户", expanded=False):
            new_sub_name = st.text_input("子账户名称", placeholder="breakout_us / swing_cn", key="strategy_lab_new_sub_name")
            new_sub_cap = st.number_input("初始资金", min_value=1000.0, value=20000.0, step=1000.0, key="strategy_lab_new_sub_cap")
            if st.button("创建子账户", key="strategy_lab_create_sub_btn"):
                created = create_paper_account(new_sub_name.strip(), float(new_sub_cap))
                if created.get('success'):
                    st.success(f"✅ 子账户已创建: {created['account_name']}")
                    st.rerun()
                else:
                    st.error(f"❌ {created.get('error', '创建失败')}")

        with st.expander("🛡️ 子账户风控设置", expanded=False):
            strategy_note = st.text_area(
                "策略说明",
                value=account_cfg.get('strategy_note', ''),
                height=80,
                key="strategy_lab_paper_strategy_note"
            )
            single_pos_limit = st.slider(
                "单票仓位上限 (%)",
                min_value=5,
                max_value=80,
                value=int(round(float(account_cfg.get('max_single_position_pct', 0.30)) * 100)),
                key="strategy_lab_paper_single_pos_limit"
            )
            max_dd_limit = st.slider(
                "最大回撤阈值 (%)",
                min_value=5,
                max_value=80,
                value=int(round(float(account_cfg.get('max_drawdown_pct', 0.20)) * 100)),
                key="strategy_lab_paper_max_dd_limit"
            )
            if st.button("保存风控设置", key="strategy_lab_save_paper_config_btn"):
                saved = update_paper_account_config(
                    selected_account,
                    strategy_note=strategy_note,
                    max_single_position_pct=single_pos_limit / 100.0,
                    max_drawdown_pct=max_dd_limit / 100.0
                )
                if saved.get('success'):
                    st.success("✅ 已保存")
                    st.rerun()
                else:
                    st.error(f"❌ {saved.get('error', '保存失败')}")
        
        # ==========================================
        # 子账户管理工具
        # ==========================================
        with st.expander("🔧 子账户工具", expanded=False):
            tool_tab1, tool_tab2, tool_tab3 = st.tabs(["📤 导出", "📥 导入", "🗑️ 删除"])
            
            with tool_tab1:
                st.markdown("**导出当前子账户数据为 JSON**")
                st.caption(f"将导出: 账户设置、持仓、交易记录")
                
                if st.button("📤 导出 JSON", key="export_account_btn"):
                    try:
                        from services.portfolio_service import export_paper_account
                        import json
                        
                        result = export_paper_account(selected_account)
                        if result.get('success'):
                            json_data = json.dumps(result['data'], indent=2, ensure_ascii=False, default=str)
                            
                            st.download_button(
                                label="💾 下载 JSON 文件",
                                data=json_data,
                                file_name=f"paper_account_{selected_account}_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json",
                                key="download_json_btn"
                            )
                            
                            positions_count = len(result['data'].get('positions', []))
                            trades_count = len(result['data'].get('trades', []))
                            st.success(f"✅ 已准备导出: {positions_count} 个持仓, {trades_count} 条交易记录")
                        else:
                            st.error(f"❌ 导出失败: {result.get('error')}")
                    except Exception as e:
                        st.error(f"导出错误: {e}")
            
            with tool_tab2:
                st.markdown("**从 JSON 文件导入子账户**")
                
                uploaded_file = st.file_uploader(
                    "选择 JSON 文件",
                    type=['json'],
                    key="import_json_uploader"
                )
                
                import_new_name = st.text_input(
                    "新账户名 (留空自动生成)",
                    placeholder="my_strategy",
                    key="import_new_name"
                )
                
                if uploaded_file is not None:
                    if st.button("📥 开始导入", key="import_account_btn"):
                        try:
                            import json
                            from services.portfolio_service import import_paper_account
                            
                            data = json.load(uploaded_file)
                            result = import_paper_account(data, import_new_name.strip() if import_new_name else None)
                            
                            if result.get('success'):
                                st.success(f"✅ 导入成功!")
                                st.info(f"新账户: **{result['account_name']}** | 持仓: {result['imported_positions']} | 交易记录: {result['imported_trades']}")
                                st.rerun()
                            else:
                                st.error(f"❌ 导入失败: {result.get('error')}")
                        except json.JSONDecodeError:
                            st.error("❌ JSON 格式错误")
                        except Exception as e:
                            st.error(f"导入错误: {e}")
            
            with tool_tab3:
                st.markdown("**删除子账户**")
                st.warning("⚠️ 删除后无法恢复，建议先导出备份")
                
                if selected_account == 'default':
                    st.info("默认账户不能删除")
                else:
                    st.markdown(f"即将删除: **{selected_account}**")
                    
                    confirm_delete = st.checkbox(
                        f"我确认要删除 {selected_account} 及其所有数据",
                        key="confirm_delete_account"
                    )
                    
                    if st.button("🗑️ 永久删除", type="primary", disabled=not confirm_delete, key="delete_account_btn"):
                        try:
                            from services.portfolio_service import delete_paper_account
                            
                            result = delete_paper_account(selected_account)
                            if result.get('success'):
                                st.success(f"✅ 已删除子账户: {selected_account}")
                                st.session_state['global_paper_account_name'] = 'default'
                                st.rerun()
                            else:
                                st.error(f"❌ 删除失败: {result.get('error')}")
                        except Exception as e:
                            st.error(f"删除错误: {e}")
        
        # 获取账户信息
        account = get_paper_account(selected_account)

        
        # 账户概览
        if account is None:
            st.warning("模拟账户未初始化")
            if st.button("初始化模拟账户"):
                from services.portfolio_service import init_paper_account
                init_paper_account()
                st.rerun()
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("初始资金", f"${account.get('initial_capital', 100000):,.0f}")
        with col2:
            st.metric("现金余额", f"${account.get('cash_balance', 0):,.0f}")
        with col3:
            st.metric("持仓市值", f"${account.get('position_value', 0):,.0f}")
        with col4:
            pnl = account.get('total_pnl', 0)
            initial = account.get('initial_capital', 100000)
            st.metric("总盈亏", f"${pnl:,.0f}",
                     delta=f"{pnl/initial*100:.1f}%" if initial > 0 else "0%",
                     delta_color="normal" if pnl >= 0 else "inverse")
        
        st.divider()

        # 默认策略组合（自动建仓）
        st.markdown("#### 🧩 默认策略组合（自动建仓）")
        with st.expander("按策略模板生成组合 + 一键等金额买入", expanded=False):
            def _pick_first(row: dict, keys: list, default=None):
                for k in keys:
                    if k in row and row.get(k) is not None:
                        return row.get(k)
                return default

            def _to_float(v, default=0.0) -> float:
                try:
                    if v is None or v == "":
                        return float(default)
                    return float(v)
                except Exception:
                    return float(default)

            def _to_bool(v) -> bool:
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return v != 0
                if isinstance(v, str):
                    s = v.strip().lower()
                    return s in ("1", "true", "yes", "y", "是")
                return False

            def _extract_signal_fields(row: dict) -> dict:
                day_blue = _to_float(_pick_first(row, ["blue_daily", "Day_BLUE", "day_blue"], 0.0), 0.0)
                week_blue = _to_float(_pick_first(row, ["blue_weekly", "Week_BLUE", "week_blue"], 0.0), 0.0)
                month_blue = _to_float(_pick_first(row, ["blue_monthly", "Month_BLUE", "month_blue"], 0.0), 0.0)
                price = _to_float(_pick_first(row, ["price", "Close", "close"], 0.0), 0.0)

                is_heima = _to_bool(_pick_first(row, ["is_heima", "Is_Heima", "heima_daily", "Heima_Daily"], False))
                week_heima = _to_bool(_pick_first(row, ["heima_weekly", "Heima_Weekly"], False))
                month_heima = _to_bool(_pick_first(row, ["heima_monthly", "Heima_Monthly"], False))

                # 兜底: Strategy 字段包含“黑马”
                strategy_text = str(_pick_first(row, ["strategy", "Strategy"], "") or "")
                if ("黑马" in strategy_text) and not (is_heima or week_heima or month_heima):
                    is_heima = True

                return {
                    "symbol": str(_pick_first(row, ["symbol", "Symbol"], "") or "").upper().strip(),
                    "price": price,
                    "day_blue": day_blue,
                    "week_blue": week_blue,
                    "month_blue": month_blue,
                    "is_heima": is_heima,
                    "week_heima": week_heima,
                    "month_heima": month_heima,
                }

            auto_col1, auto_col2, auto_col3, auto_col4 = st.columns(4)
            with auto_col1:
                auto_market = st.selectbox("市场", ["US", "CN"], key="auto_basket_market")
            with auto_col2:
                auto_min_blue = st.slider("BLUE阈值", min_value=60, max_value=160, value=100, step=5, key="auto_basket_min_blue")
            with auto_col3:
                auto_top_n = st.number_input("TopN(非全买策略)", min_value=5, max_value=60, value=20, step=1, key="auto_basket_top_n")
            with auto_col4:
                auto_deploy_pct = st.slider("资金使用率(%)", min_value=10, max_value=100, value=90, step=5, key="auto_basket_deploy_pct")

            auto_col5, auto_col6, auto_col7, auto_col8 = st.columns(4)
            with auto_col5:
                auto_seed_cap = st.number_input("新账户初始资金($)", min_value=1000.0, value=20000.0, step=1000.0, key="auto_basket_seed_cap")
            with auto_col6:
                auto_full_cap = st.number_input("全买策略上限只数", min_value=20, max_value=300, value=120, step=10, key="auto_basket_full_cap")
            with auto_col7:
                auto_reset_before_buy = st.checkbox("执行前重置目标子账户", value=False, key="auto_basket_reset_before_buy")
            with auto_col8:
                auto_track_days = st.slider("追踪统计天数", min_value=30, max_value=720, value=180, step=30, key="auto_basket_track_days")

            _set_active_market(auto_market)

            strategy_defs = [
                {
                    "id": "daily_equal",
                    "name": "每日组合等权",
                    "desc": "日BLUE>=阈值，按综合分排序取TopN，等金额买入",
                    "account_tag": "daily_equal",
                    "full_buy": False,
                },
                {
                    "id": "month_heima_all",
                    "name": "月黑马全买",
                    "desc": "月线黑马/月级黑马信号，满足即入池（受上限只数保护）",
                    "account_tag": "month_heima_all",
                    "full_buy": True,
                },
                {
                    "id": "week_heima_all",
                    "name": "周黑马全买",
                    "desc": "周线黑马信号，满足即入池（受上限只数保护）",
                    "account_tag": "week_heima_all",
                    "full_buy": True,
                },
                {
                    "id": "day_week_resonance",
                    "name": "日周共振",
                    "desc": "日BLUE+周BLUE 同时过阈值，取TopN",
                    "account_tag": "day_week_res",
                    "full_buy": False,
                },
                {
                    "id": "core_resonance",
                    "name": "核心共振",
                    "desc": "日周过阈值且(月BLUE过阈值或任意黑马)",
                    "account_tag": "core_res",
                    "full_buy": False,
                },
            ]

            def _strategy_match(rule_id: str, f: dict) -> bool:
                d = f["day_blue"]
                w = f["week_blue"]
                m = f["month_blue"]
                h_any = f["is_heima"] or f["week_heima"] or f["month_heima"]

                if rule_id == "daily_equal":
                    return d >= auto_min_blue
                if rule_id == "month_heima_all":
                    return f["month_heima"] or (m >= auto_min_blue and h_any)
                if rule_id == "week_heima_all":
                    return f["week_heima"] or (w >= auto_min_blue and h_any)
                if rule_id == "day_week_resonance":
                    return d >= auto_min_blue and w >= auto_min_blue
                if rule_id == "core_resonance":
                    return d >= auto_min_blue and w >= auto_min_blue and (m >= auto_min_blue or h_any)
                return False

            def _strategy_score(f: dict) -> float:
                base = f["day_blue"] * 0.45 + f["week_blue"] * 0.35 + f["month_blue"] * 0.20
                bonus = 0.0
                if f["is_heima"]:
                    bonus += 25.0
                if f["week_heima"]:
                    bonus += 20.0
                if f["month_heima"]:
                    bonus += 20.0
                return base + bonus

            auto_dates = _cached_scanned_dates(market=auto_market)
            if not auto_dates:
                st.warning(f"暂无 {auto_market} 扫描数据，无法生成默认组合。")
            else:
                auto_latest_date = auto_dates[0]
                auto_rows = _cached_scan_results(scan_date=auto_latest_date, market=auto_market, limit=2000) or []
                st.caption(f"使用扫描日: `{auto_latest_date}` | 样本数: {len(auto_rows)}")

                basket_results = {}
                for sd in strategy_defs:
                    sym_map = {}
                    for r in auto_rows:
                        f = _extract_signal_fields(r)
                        sym = f["symbol"]
                        if not sym or f["price"] <= 0:
                            continue
                        if not _strategy_match(sd["id"], f):
                            continue
                        score = _strategy_score(f)
                        prev = sym_map.get(sym)
                        if (prev is None) or (score > prev["score"]):
                            sym_map[sym] = {
                                "symbol": sym,
                                "price": f["price"],
                                "day_blue": f["day_blue"],
                                "week_blue": f["week_blue"],
                                "month_blue": f["month_blue"],
                                "is_heima": f["is_heima"] or f["week_heima"] or f["month_heima"],
                                "score": score,
                            }

                    picks = sorted(sym_map.values(), key=lambda x: x["score"], reverse=True)
                    if sd["full_buy"]:
                        picks = picks[: int(auto_full_cap)]
                    else:
                        picks = picks[: int(auto_top_n)]
                    basket_results[sd["id"]] = picks

                st.markdown("**策略候选与执行**")
                for sd in strategy_defs:
                    picks = basket_results.get(sd["id"], [])
                    account_name = f"auto_{auto_market.lower()}_{sd['account_tag']}"
                    header = f"{sd['name']} | {len(picks)} 只 | 子账户: `{account_name}`"
                    with st.expander(header, expanded=False):
                        st.caption(sd["desc"])
                        if picks:
                            preview_df = pd.DataFrame(picks)
                            preview_df["黑马"] = preview_df["is_heima"].map(lambda x: "是" if x else "否")
                            show_cols = ["symbol", "price", "day_blue", "week_blue", "month_blue", "黑马", "score"]
                            rename_cols = {
                                "symbol": "代码",
                                "price": "价格",
                                "day_blue": "日BLUE",
                                "week_blue": "周BLUE",
                                "month_blue": "月BLUE",
                                "score": "综合分",
                            }
                            st.dataframe(
                                preview_df[show_cols].rename(columns=rename_cols),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("当前条件无候选股票。")

                        if st.button(f"🚀 执行买入: {sd['name']}", key=f"run_auto_basket_{sd['id']}_{auto_market}"):
                            if not picks:
                                st.warning("无候选股票，跳过执行。")
                            else:
                                created = create_paper_account(account_name, float(auto_seed_cap))
                                if (not created.get("success")) and ("已存在" not in str(created.get("error", ""))):
                                    st.error(f"创建子账户失败: {created.get('error')}")
                                else:
                                    if auto_reset_before_buy:
                                        reset_paper_account(account_name)

                                    target_acc = get_paper_account(account_name)
                                    cash_balance = float((target_acc or {}).get("cash_balance", 0.0) or 0.0)
                                    deploy_cap = cash_balance * (float(auto_deploy_pct) / 100.0)
                                    if deploy_cap <= 0:
                                        st.error("可用资金不足，无法执行。")
                                    else:
                                        per_stock_budget = deploy_cap / max(len(picks), 1)
                                        success_cnt = 0
                                        fail_msgs = []
                                        skip_cnt = 0
                                        for item in picks:
                                            px = float(item.get("price", 0.0) or 0.0)
                                            if px <= 0:
                                                skip_cnt += 1
                                                continue
                                            qty = int(per_stock_budget / px)
                                            if qty < 1:
                                                skip_cnt += 1
                                                continue
                                            ret = paper_buy(item["symbol"], qty, px, auto_market, account_name)
                                            if ret.get("success"):
                                                success_cnt += 1
                                            else:
                                                fail_msgs.append(f"{item['symbol']}: {ret.get('error', '失败')}")
                                        st.success(
                                            f"执行完成: 成功 {success_cnt} | 跳过 {skip_cnt} | 失败 {len(fail_msgs)}"
                                        )
                                        if fail_msgs:
                                            st.warning(" ; ".join(fail_msgs[:5]))
                                        st.rerun()

                st.markdown("**默认组合绩效对比**")
                tracking_rows_all = []
                try:
                    from services.candidate_tracking_service import get_candidate_tracking_rows as get_candidate_tracking_rows_for_perf
                    tracking_rows_all = get_candidate_tracking_rows_for_perf(market=auto_market, days_back=int(auto_track_days)) or []
                except Exception as e:
                    st.caption(f"⚠️ 候选追踪统计不可用: {e}")

                perf_rows = []
                for sd in strategy_defs:
                    acc_name = f"auto_{auto_market.lower()}_{sd['account_tag']}"
                    if acc_name not in sub_names:
                        continue
                    perf = get_paper_account_performance(acc_name)
                    acc_data = get_paper_account(acc_name) or {}
                    pos = acc_data.get("positions", []) or []
                    open_winners = sum(1 for p in pos if float(p.get("unrealized_pnl", 0.0) or 0.0) > 0)
                    open_win_rate = (open_winners / len(pos) * 100.0) if pos else 0.0

                    matched_track_rows = []
                    if tracking_rows_all:
                        for tr in tracking_rows_all:
                            tf = _extract_signal_fields(tr)
                            if _strategy_match(sd["id"], tf):
                                matched_track_rows.append(tr)
                    track_total = len(matched_track_rows)
                    if track_total > 0:
                        track_wins = sum(1 for r in matched_track_rows if float(r.get("pnl_pct") or 0) > 0)
                        track_win_rate = track_wins / track_total * 100.0
                        track_avg_pnl = float(np.mean([float(r.get("pnl_pct") or 0) for r in matched_track_rows]))
                        d2p_vals = [int(r["first_positive_day"]) for r in matched_track_rows if r.get("first_positive_day") is not None]
                        track_median_d2p = int(np.median(d2p_vals)) if d2p_vals else None
                    else:
                        track_win_rate = None
                        track_avg_pnl = None
                        track_median_d2p = None

                    perf_rows.append({
                        "策略": sd["name"],
                        "子账户": acc_name,
                        "总收益率": f"{float(perf.get('total_return_pct', 0.0)):+.2f}%",
                        "胜率(已平仓)": f"{float(perf.get('win_rate_pct', 0.0)):.1f}%",
                        "赢面(当前持仓)": f"{open_win_rate:.1f}%",
                        "追踪样本": track_total,
                        "追踪胜率": f"{track_win_rate:.1f}%" if track_win_rate is not None else "-",
                        "追踪均收": f"{track_avg_pnl:+.2f}%" if track_avg_pnl is not None else "-",
                        "转正中位天": f"{track_median_d2p}天" if track_median_d2p is not None else "-",
                        "盈亏比": "∞" if perf.get("profit_factor") == float("inf") else f"{float(perf.get('profit_factor', 0.0)):.2f}",
                        "总盈亏": f"${float(perf.get('total_pnl', 0.0)):+,.2f}",
                        "持仓数": len(pos),
                        "今日候选": len(basket_results.get(sd["id"], [])),
                    })

                if perf_rows:
                    perf_df = pd.DataFrame(perf_rows)
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)

                    if st.button("📣 推送默认组合绩效", key=f"push_auto_basket_perf_{auto_market}"):
                        try:
                            from services.notification import NotificationManager

                            top_rows = perf_df.head(5).to_dict("records")
                            lines = [
                                f"*🧩 默认组合绩效 | {auto_market}*",
                                f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                f"追踪窗口: 最近 {int(auto_track_days)} 天",
                                "",
                            ]
                            for row in top_rows:
                                lines.append(
                                    f"- {row['策略']}: 总收益 {row['总收益率']} | 追踪胜率 {row['追踪胜率']} | "
                                    f"追踪均收 {row['追踪均收']} | 样本 {row['追踪样本']} | 子账户 `{row['子账户']}`"
                                )
                            lines.append("")
                            lines.append("仅供研究，不构成投资建议。")
                            msg = "\n".join(lines)

                            nm = NotificationManager()
                            tg_ok = nm.send_telegram(msg) if nm.telegram_token else False
                            wc_ok = nm.send_wecom(msg, msg_type="markdown") if nm.wecom_webhook else False
                            wx_ok = nm.send_wxpusher(
                                title=f"Coral Creek 默认组合绩效 {auto_market}",
                                content=msg,
                            ) if nm.wxpusher_app_token else False
                            bark_ok = nm.send_bark(
                                title=f"默认组合绩效 {auto_market}",
                                content=msg,
                            ) if nm.bark_url else False

                            st.success(
                                f"推送完成 | telegram={tg_ok}, wecom={wc_ok}, wxpusher={wx_ok}, bark={bark_ok}"
                            )
                        except Exception as e:
                            st.error(f"推送失败: {e}")
                else:
                    st.info("默认组合子账户尚未创建。先点上方任一策略“执行买入”。")
        
        # 交易面板
        trade_col1, trade_col2 = st.columns(2)
        
        with trade_col1:
            st.markdown("#### 🟢 买入")
            with st.form("paper_buy_form"):
                symbol = st.text_input("股票代码", placeholder="AAPL")
                shares = st.number_input("股数", min_value=1, value=10)
                price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0)
                market = st.selectbox("市场", ["US", "CN"])
                
                if st.form_submit_button("买入", type="primary"):
                    if symbol:
                        result = paper_buy(symbol.upper(), shares, price if price > 0 else None, market, selected_account)
                        if result.get('success'):
                            st.success(f"✅ 买入成功: {shares} 股 {symbol}")
                            st.rerun()
                        else:
                            st.error(f"❌ 买入失败: {result.get('error')}")
        
        with trade_col2:
            st.markdown("#### 🔴 卖出")
            positions = account.get('positions', [])
            if positions:
                with st.form("paper_sell_form"):
                    pos_options = [f"{p['symbol']} ({p['shares']}股)" for p in positions]
                    selected = st.selectbox("选择持仓", pos_options)
                    sell_shares = st.number_input("卖出股数", min_value=1, value=1)
                    sell_price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0)
                    
                    if st.form_submit_button("卖出", type="primary"):
                        symbol = selected.split(" ")[0]
                        selected_pos = next((p for p in positions if p.get('symbol') == symbol), {})
                        sell_market = selected_pos.get('market', 'US')
                        result = paper_sell(
                            symbol,
                            sell_shares,
                            sell_price if sell_price > 0 else None,
                            sell_market,
                            selected_account
                        )
                        if result.get('success'):
                            st.success(f"✅ 卖出成功: {sell_shares} 股 {symbol}")
                            st.rerun()
                        else:
                            st.error(f"❌ 卖出失败: {result.get('error')}")
            else:
                st.info("暂无持仓")
        
        # 权益曲线
        equity_curve = get_paper_equity_curve(selected_account)
        if not equity_curve.empty and len(equity_curve) > 1:
            st.markdown("#### 📈 权益曲线")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['total_equity'],
                mode='lines+markers',
                name='总权益'
            ))
            initial_cap = account.get('initial_capital', 100000)
            fig.add_hline(y=initial_cap, line_dash="dash", 
                         annotation_text=f"初始资金 ${initial_cap:,.0f}")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # 子账户策略绩效对比
        st.markdown("#### 🏁 子账户策略绩效对比")
        perf_rows = get_all_paper_accounts_performance()
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            if not perf_df.empty:
                show_df = perf_df.copy()
                show_df['总收益率'] = show_df['total_return_pct'].map(lambda x: f"{x:+.2f}%")
                show_df['最大回撤'] = show_df['max_drawdown_pct'].map(lambda x: f"{x:.2f}%")
                show_df['胜率'] = show_df['win_rate_pct'].map(lambda x: f"{x:.1f}%")
                show_df['已平仓笔数'] = show_df['closed_trades']
                show_df['总交易数'] = show_df['total_trades']
                show_df['盈亏'] = show_df['total_pnl'].map(lambda x: f"${x:+,.2f}")
                show_df['因子'] = show_df['profit_factor'].map(
                    lambda x: "∞" if x == float('inf') else f"{x:.2f}"
                )
                show_df = show_df.rename(columns={'account_name': '子账户'})
                st.dataframe(
                    show_df[['子账户', '总收益率', '最大回撤', '胜率', '已平仓笔数', '总交易数', '因子', '盈亏']],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("暂无可对比的子账户绩效")

        # 子账户权益曲线对比
        st.markdown("#### 📊 子账户权益曲线对比")
        
        col_curve_sel, col_curve_opt = st.columns([3, 1])
        with col_curve_sel:
            curve_options = [a for a in sub_names]
            # 默认选中当前账户和表现最好的账户（如果有）
            default_compare = [selected_account]
            if len(sub_names) > 1 and sub_names[0] != selected_account:
                default_compare.append(sub_names[0])
                
            compare_accounts = st.multiselect(
                "选择对比子账户",
                options=curve_options,
                default=default_compare[:3], # 最多默认选3个
                key="paper_compare_accounts"
            )
        
        with col_curve_opt:
            normalize_curve = st.checkbox("归一化 (起点=100)", value=True, help="将所有账户起始资金设为100，便于对比收益率走势")

        if compare_accounts:
            try:
                from services.portfolio_service import get_multi_account_equity_curves
                df_curves = get_multi_account_equity_curves(compare_accounts, normalize=normalize_curve)
                
                if not df_curves.empty:
                    import plotly.graph_objects as go
                    fig_compare = go.Figure()
                    
                    for col in df_curves.columns:
                        if col == 'date': 
                            continue
                            
                        fig_compare.add_trace(go.Scatter(
                            x=df_curves['date'],
                            y=df_curves[col],
                            mode='lines',
                            name=col,
                            hovertemplate='%{y:.2f}'
                        ))
                    
                    y_title = "相对收益 (起点=100)" if normalize_curve else "总权益 ($)"
                    
                    fig_compare.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title="日期",
                        yaxis_title=y_title,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode="x unified"
                    )
                    
                    # 如果是归一化，画一条 100 的基准线
                    if normalize_curve:
                        fig_compare.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
                        
                    st.plotly_chart(fig_compare, use_container_width=True)
                else:
                    st.info("所选子账户暂无足够数据生成曲线")
            except Exception as e:
                st.error(f"生成图表失败: {e}")
        else:
            st.info("请选择至少一个子账户进行对比")
        
        # 重置按钮
        if st.button("🔄 重置模拟账户", type="secondary"):
            reset_paper_account(selected_account)
            st.success("账户已重置")
            st.rerun()
            
    except Exception as e:
        st.warning(f"模拟交易模块加载失败: {e}")


def render_strategy_lab_page():
    """🧪 策略实验室 - 合并: 回测 + 研究工具 + 模拟盘"""
    st.header("🧪 策略实验室")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 策略回测", 
        "🎛️ 策略组合", 
        "💰 模拟盘", 
        "🔬 因子研究", 
        "📐 组合优化"
    ])
    
    with tab1:
        render_backtest_page()
    
    with tab2:
        render_strategy_component_page()
    
    with tab3:
        render_paper_trading_page()
    
    with tab4:
        render_research_page()
    
    with tab5:
        render_portfolio_optimizer_page()


def render_strategy_component_page():
    """🎛️ 策略组合 - 自由组合买卖条件回测"""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    st.subheader("🎛️ 策略组合回测")
    st.markdown("自由组合买入/卖出条件，测试策略表现")
    
    # 尝试导入策略组件
    try:
        from strategies.strategy_components import (
            StrategyBuilder,
            BUY_CONDITIONS,
            SELL_CONDITIONS,
        )
        from indicator_utils import (
            calculate_blue_signal_series, 
            calculate_heima_signal_series, 
            calculate_kdj_series
        )
    except ImportError as e:
        st.error(f"无法加载策略组件: {e}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 买入条件 (满足任一)")
        buy_options = {
            'blue_heima': 'BLUE≥100 + 黑马共振',
            'strong_blue': '强BLUE≥150 + 黑马',
            'double_blue': '日周双BLUE≥150',
            'bottom_peak': '底部筹码顶格峰',
            'blue_only': '超强BLUE≥200',
            'heima_only': '纯黑马/掘地',
        }
        selected_buy = []
        for key, label in buy_options.items():
            if st.checkbox(label, value=(key == 'blue_heima'), key=f"comp_buy_{key}"):
                selected_buy.append(key)
    
    with col2:
        st.markdown("#### 🔴 卖出条件 (满足任一)")
        sell_options = {
            'kdj_overbought': 'KDJ J>90 超买',
            'chip_distribution': '筹码顶部堆积',
            'chip_with_ma': '跌破MA5+筹码异常',
            'ma_break': '跌破MA5',
            'profit_target_20': '止盈20%',
            'stop_loss_8': '止损-8%',
        }
        selected_sell = []
        for key, label in sell_options.items():
            default = key in ['kdj_overbought', 'chip_distribution']
            if st.checkbox(label, value=default, key=f"comp_sell_{key}"):
                selected_sell.append(key)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols_input = st.text_input("股票代码 (逗号分隔)", "AAPL, MSFT, GOOGL")
    with col2:
        days = st.slider("回测天数", 180, 1095, 730)
    with col3:
        run_btn = st.button("🚀 运行回测", type="primary", key="run_custom_backtest")
    
    if run_btn and selected_buy and selected_sell:
        symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
        
        strategy = StrategyBuilder("自定义策略")
        for cond in selected_buy:
            strategy.add_buy_condition(cond)
        for cond in selected_sell:
            strategy.add_sell_condition(cond)
        
        st.info(f"**买入**: {', '.join([buy_options[k] for k in selected_buy])}")
        st.info(f"**卖出**: {', '.join([sell_options[k] for k in selected_sell])}")
        
        progress = st.progress(0)
        results = []
        
        for idx, symbol in enumerate(symbols):
            progress.progress((idx + 1) / len(symbols))
            try:
                df = get_stock_data(symbol, 'US', days=days)
                if df is None or len(df) < 100:
                    continue
                
                # 简化回测
                data = _prepare_backtest_data(df)
                result = _run_simple_backtest(df, data, strategy)
                result['symbol'] = symbol
                results.append(result)
            except Exception as e:
                st.warning(f"{symbol}: {e}")
        
        progress.empty()
        
        if results:
            df_results = pd.DataFrame([{
                '股票': r['symbol'],
                '年化收益%': f"{r['annual_return']:.1f}%",
                '最大回撤%': f"{r['max_drawdown']:.1f}%",
                '胜率%': f"{r['win_rate']:.1f}%",
                '交易次数': r['trades']
            } for r in results])
            
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("平均年化", f"{np.mean([r['annual_return'] for r in results]):.1f}%")
            col2.metric("平均胜率", f"{np.mean([r['win_rate'] for r in results]):.1f}%")
            col3.metric("平均回撤", f"{np.mean([r['max_drawdown'] for r in results]):.1f}%")
    
    elif run_btn:
        st.warning("请至少选择一个买入条件和一个卖出条件")


def _prepare_backtest_data(df):
    """准备回测数据"""
    from indicator_utils import (
        calculate_blue_signal_series, 
        calculate_heima_signal_series, 
        calculate_kdj_series
    )
    import numpy as np
    
    blue = calculate_blue_signal_series(
        df['Open'].values, df['High'].values,
        df['Low'].values, df['Close'].values
    )
    heima, juedi = calculate_heima_signal_series(
        df['High'].values, df['Low'].values,
        df['Close'].values, df['Open'].values
    )
    _, _, j = calculate_kdj_series(
        df['High'].values, df['Low'].values, df['Close'].values
    )
    
    # 周线
    df_weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    if len(df_weekly) >= 5:
        week_blue = calculate_blue_signal_series(
            df_weekly['Open'].values, df_weekly['High'].values,
            df_weekly['Low'].values, df_weekly['Close'].values
        )
        week_heima, week_juedi = calculate_heima_signal_series(
            df_weekly['High'].values, df_weekly['Low'].values,
            df_weekly['Close'].values, df_weekly['Open'].values
        )
        df_weekly['Week_BLUE'] = week_blue
        df_weekly['Week_Heima'] = week_heima
        df_weekly['Week_Juedi'] = week_juedi
        
        week_blue_ref = df_weekly['Week_BLUE'].shift(1).reindex(
            df.index, method='ffill'
        ).fillna(0).values
        week_heima_ref = df_weekly['Week_Heima'].shift(1).reindex(
            df.index, method='ffill'
        ).fillna(False).values
        week_juedi_ref = df_weekly['Week_Juedi'].shift(1).reindex(
            df.index, method='ffill'
        ).fillna(False).values
    else:
        week_blue_ref = np.zeros(len(df))
        week_heima_ref = np.zeros(len(df), dtype=bool)
        week_juedi_ref = np.zeros(len(df), dtype=bool)
    
    ma5 = df['Close'].rolling(5).mean().values
    
    return {
        'blue': blue, 'heima': heima, 'juedi': juedi, 'kdj_j': j,
        'week_blue': week_blue_ref, 'week_heima': week_heima_ref, 
        'week_juedi': week_juedi_ref, 'ma5': ma5,
        'close': df['Close'].values, 'low': df['Low'].values,
    }


def _run_simple_backtest(df, data, strategy, initial_capital=100000):
    """运行简化回测"""
    import numpy as np
    
    cash = initial_capital
    shares = 0
    position = 0
    trades = 0
    wins = 0
    entry_price = 0
    equity = [initial_capital]
    
    for i in range(50, len(df) - 1):
        close = data['close'][i]
        next_open = df['Open'].iloc[i + 1]
        
        if position == 1:
            strategy.update_peak_price(close)
            should_sell, reason = strategy.check_sell(data, i, df)
            if should_sell:
                revenue = shares * close
                pnl = revenue - entry_price * shares
                if pnl > 0:
                    wins += 1
                trades += 1
                cash += revenue
                shares = 0
                position = 0
                strategy.reset_position()
        
        elif position == 0:
            should_buy, reason = strategy.check_buy(data, i, df)
            if should_buy and cash > 0:
                shares = int(cash / next_open)
                if shares > 0:
                    cash -= shares * next_open
                    position = 1
                    entry_price = next_open
                    strategy.set_entry_price(next_open)
        
        equity.append(cash + shares * close)
    
    equity = np.array(equity)
    final = equity[-1]
    days = len(df)
    
    total_return = (final / initial_capital - 1) * 100
    annual_return = ((final / initial_capital) ** (252 / days) - 1) * 100
    
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_drawdown = np.max(drawdown)
    
    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    return {
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades,
    }


def render_paper_trading_page():
    """💰 模拟盘交易页面"""
    st.subheader("💰 模拟盘交易")

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    def _show_trade_error(err: Exception) -> None:
        msg = str(err)
        if "风控拦截" in msg:
            st.warning(f"🛡️ {msg}")
        else:
            st.error(f"❌ 下单失败: {msg}")

    def _upsert_env_values(file_path: str, values: dict) -> None:
        lines = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

        updated = {k: False for k in values}
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                new_lines.append(line)
                continue

            key = stripped.split("=", 1)[0].strip()
            if key in values:
                new_lines.append(f"{key}={values[key]}")
                updated[key] = True
            else:
                new_lines.append(line)

        for key, done in updated.items():
            if not done:
                new_lines.append(f"{key}={values[key]}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

    def _resolve_alpaca_keys():
        """优先读取环境变量，其次读取 Streamlit secrets（含 [alpaca] 分组）"""
        api = os.environ.get('ALPACA_API_KEY')
        secret = os.environ.get('ALPACA_SECRET_KEY')
        if api and secret:
            return api, secret

        try:
            if hasattr(st, "secrets"):
                # 顶层键
                api = api or st.secrets.get("ALPACA_API_KEY") or st.secrets.get("alpaca_api_key")
                secret = secret or st.secrets.get("ALPACA_SECRET_KEY") or st.secrets.get("alpaca_secret_key")

                # [alpaca] 分组键
                alpaca_group = st.secrets.get("alpaca")
                if isinstance(alpaca_group, dict):
                    api = api or alpaca_group.get("api_key") or alpaca_group.get("ALPACA_API_KEY")
                    secret = secret or alpaca_group.get("secret_key") or alpaca_group.get("ALPACA_SECRET_KEY")
        except Exception:
            pass

        return api, secret
    
    # 检查 Alpaca SDK
    try:
        from execution.alpaca_trader import (
            AlpacaTrader, 
            SignalTrader,
            ALPACA_SDK_AVAILABLE
        )
    except ImportError:
        ALPACA_SDK_AVAILABLE = False
    
    if not ALPACA_SDK_AVAILABLE:
        st.warning("⚠️ Alpaca SDK 未安装")
        st.code("pip install alpaca-py", language="bash")
        
        st.markdown("""
        ### 设置步骤
        
        1. **注册 Alpaca 账号** (免费): [https://alpaca.markets/](https://alpaca.markets/)
        
        2. **获取 API Keys**:
           - 登录后点击 "Paper Trading"
           - 点击 "Your API Keys"
           - 复制 API Key 和 Secret Key
        
        3. **配置环境变量** (在 `.env` 文件中添加):
           ```
           ALPACA_API_KEY=your_api_key_here
           ALPACA_SECRET_KEY=your_secret_key_here
           ALPACA_PAPER=true
           ```
        
        4. **安装 SDK**: `pip install alpaca-py`
        
        5. **重启应用**
        """)
        return
    
    # 检查 API Keys
    import os
    api_key, secret_key = _resolve_alpaca_keys()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    if not api_key or not secret_key:
        st.warning("⚠️ 未配置 Alpaca API Keys")
        
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("API Key", type="password", key="alpaca_api")
        with col2:
            secret_key = st.text_input("Secret Key", type="password", key="alpaca_secret")
        
        if not api_key or not secret_key:
            st.info("请输入 Alpaca API Keys 或在 .env 文件中配置")
            return

    enable_hard_risk_guards = os.environ.get("ALPACA_ENABLE_HARD_RISK_GUARDS", "true").lower() == "true"
    max_single_position_pct = _env_float("ALPACA_MAX_SINGLE_POSITION_PCT", 0.20)
    max_daily_loss_pct = _env_float("ALPACA_MAX_DAILY_LOSS_PCT", 0.03)
    max_portfolio_drawdown_pct = _env_float("ALPACA_MAX_PORTFOLIO_DRAWDOWN_PCT", 0.15)
    
    # 连接
    try:
        trader = AlpacaTrader(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
            enable_hard_risk_guards=enable_hard_risk_guards,
            max_single_position_pct=max_single_position_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            max_portfolio_drawdown_pct=max_portfolio_drawdown_pct
        )
        account = trader.get_account()
    except Exception as e:
        st.error(f"❌ 连接失败: {e}")
        return
    
    # 账户信息
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("账户净值", f"${account.equity:,.2f}")
    col2.metric("可用现金", f"${account.cash:,.2f}")
    col3.metric("购买力", f"${account.buying_power:,.2f}")
    
    market = trader.get_market_hours()
    status = "🟢 开盘中" if market['is_open'] else "🔴 休市"
    col4.metric("市场状态", status)

    st.caption("当前生效风控参数")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    risk_col1.metric("硬风控", "开启" if enable_hard_risk_guards else "关闭")
    risk_col2.metric("单票上限", f"{max_single_position_pct * 100:.1f}%")
    risk_col3.metric("日亏损上限", f"{max_daily_loss_pct * 100:.1f}%")
    risk_col4.metric("回撤上限", f"{max_portfolio_drawdown_pct * 100:.1f}%")

    with st.expander("🛡️ 风控参数（执行层）", expanded=False):
        risk_enable = st.checkbox("启用硬风控", value=enable_hard_risk_guards, key="alpaca_risk_enable_main")
        risk_single = st.slider(
            "单票最大仓位 (%)", min_value=5, max_value=50,
            value=int(round(max_single_position_pct * 100)),
            key="alpaca_risk_single_main"
        )
        risk_daily = st.slider(
            "当日最大亏损 (%)", min_value=1, max_value=20,
            value=int(round(max_daily_loss_pct * 100)),
            key="alpaca_risk_daily_main"
        )
        risk_dd = st.slider(
            "组合最大回撤 (%)", min_value=5, max_value=50,
            value=int(round(max_portfolio_drawdown_pct * 100)),
            key="alpaca_risk_dd_main"
        )

        if st.button("💾 保存风控参数", key="alpaca_risk_save_main"):
            updates = {
                "ALPACA_ENABLE_HARD_RISK_GUARDS": str(risk_enable).lower(),
                "ALPACA_MAX_SINGLE_POSITION_PCT": f"{risk_single / 100:.4f}",
                "ALPACA_MAX_DAILY_LOSS_PCT": f"{risk_daily / 100:.4f}",
                "ALPACA_MAX_PORTFOLIO_DRAWDOWN_PCT": f"{risk_dd / 100:.4f}",
            }
            _upsert_env_values(env_path, updates)
            for k, v in updates.items():
                os.environ[k] = v
            st.success("✅ 风控参数已保存，重新加载后生效")
            st.rerun()
    
    st.markdown("---")
    
    # 持仓
    st.markdown("#### 当前持仓")
    positions = trader.get_positions()
    
    if not positions:
        st.info("暂无持仓")
    else:
        pos_data = []
        total_pnl = 0
        for pos in positions:
            total_pnl += pos.unrealized_pl
            pos_data.append({
                '股票': pos.symbol,
                '数量': int(pos.qty),
                '成本价': f"${pos.avg_entry_price:.2f}",
                '现价': f"${pos.current_price:.2f}",
                '盈亏%': f"{pos.unrealized_plpc:+.2f}%"
            })
        
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        
        color = "green" if total_pnl >= 0 else "red"
        st.markdown(f"**总浮动盈亏:** <span style='color:{color}'>${total_pnl:+,.2f}</span>", 
                    unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 快速下单
    st.markdown("#### 快速下单")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol = st.text_input("股票代码", "AAPL", key="trade_symbol").upper()
    with col2:
        qty = st.number_input("数量", min_value=1, value=10, key="trade_qty")
    with col3:
        if st.button("🟢 买入", type="primary"):
            try:
                order = trader.buy_market(symbol, qty)
                st.success(f"✅ 买入订单已提交: {order['id'][:8]}...")
            except Exception as e:
                _show_trade_error(e)
    with col4:
        if st.button("🔴 卖出", type="secondary"):
            try:
                order = trader.sell_market(symbol, qty)
                st.success(f"✅ 卖出订单已提交: {order['id'][:8]}...")
            except Exception as e:
                _show_trade_error(e)

    st.markdown("---")

    # 订单管理
    st.markdown("#### 订单管理")
    order_status = st.radio(
        "订单状态",
        ["open", "closed", "all"],
        horizontal=True,
        format_func=lambda x: {"open": "待成交", "closed": "已结束", "all": "全部"}[x],
        key="alpaca_order_status_main"
    )

    try:
        orders = trader.get_orders(order_status)
    except Exception as e:
        st.error(f"❌ 获取订单失败: {e}")
        orders = []

    if not orders:
        label_map = {"open": "待成交", "closed": "已结束", "all": ""}
        st.info(f"暂无{label_map[order_status]}订单")
    else:
        stale_threshold_min = 30
        if order_status == "open":
            stale_threshold_min = st.number_input(
                "挂单超时阈值 (分钟)",
                min_value=5,
                max_value=240,
                value=30,
                step=5,
                key="alpaca_open_order_stale_threshold_main"
            )

        order_data = []
        stale_open_orders = []
        for order in orders:
            created_at = order.get("created_at") or ""
            submitted_at = order.get("submitted_at") or created_at
            age_min = None
            if order_status == "open" and submitted_at:
                ts = pd.to_datetime(submitted_at, utc=True, errors="coerce")
                if not pd.isna(ts):
                    age_min = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60.0)
                    if age_min >= stale_threshold_min:
                        stale_open_orders.append(order)

            order_data.append({
                "订单ID": str(order.get("id", ""))[:8] + "...",
                "股票": order.get("symbol", ""),
                "方向": "买入" if order.get("side") == "buy" else "卖出",
                "类型": order.get("type", ""),
                "数量": order.get("qty", ""),
                "已成交": order.get("filled_qty", 0),
                "状态": order.get("status", ""),
                "挂单时长": f"{age_min:.1f}m" if age_min is not None else "-",
                "均价": f"${float(order.get('filled_avg_price')):.2f}" if order.get("filled_avg_price") else "-",
                "创建时间": created_at[:19] if created_at else ""
            })

        st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

        if order_status == "open":
            if stale_open_orders:
                stale_symbols = ", ".join(sorted({o.get("symbol", "") for o in stale_open_orders if o.get("symbol")}))
                st.warning(
                    f"⚠️ 发现 {len(stale_open_orders)} 笔挂单超过 {int(stale_threshold_min)} 分钟"
                    + (f"（{stale_symbols}）" if stale_symbols else "")
                )
            st.caption("可对待成交订单执行撤单")
            col_o1, col_o2, col_o3 = st.columns([3, 1, 1])
            with col_o1:
                selected_open_order = st.selectbox(
                    "选择订单",
                    options=orders,
                    format_func=lambda x: (
                        f"{x.get('symbol', '')} | {x.get('side', '')} {x.get('qty', '')}股"
                        f" | 状态: {x.get('status', '')}"
                    ),
                    key="alpaca_open_order_select_main"
                )
            with col_o2:
                if st.button("撤销订单", key="alpaca_cancel_order_btn_main"):
                    ok = trader.cancel_order(selected_open_order["id"])
                    if ok:
                        st.success("✅ 订单已撤销")
                        st.rerun()
                    else:
                        st.error("❌ 撤销失败")
            with col_o3:
                if st.button("全部撤单", key="alpaca_cancel_all_orders_btn_main"):
                    ok = trader.cancel_all_orders()
                    if ok:
                        st.success("✅ 已提交全部撤单")
                        st.rerun()
                    else:
                        st.error("❌ 全部撤单失败")

    # 轻量成交质量面板（近 50 条已结束订单）
    with st.expander("📈 成交质量（简版）", expanded=False):
        try:
            closed_orders = trader.get_orders("closed")[:50]
        except Exception as e:
            st.warning(f"获取成交数据失败: {e}")
            closed_orders = []

        if not closed_orders:
            st.info("暂无已结束订单，成交后这里会显示质量指标。")
        else:
            filled_orders = [o for o in closed_orders if o.get("status") == "filled"]

            def _fill_minutes(order: dict):
                submitted_at = order.get("submitted_at") or order.get("created_at")
                filled_at = order.get("filled_at")
                if not submitted_at or not filled_at:
                    return None
                t_submit = pd.to_datetime(submitted_at, utc=True, errors="coerce")
                t_fill = pd.to_datetime(filled_at, utc=True, errors="coerce")
                if pd.isna(t_submit) or pd.isna(t_fill):
                    return None
                minutes = (t_fill - t_submit).total_seconds() / 60.0
                return minutes if minutes >= 0 else None

            def _slippage_pct(order: dict):
                if order.get("status") != "filled":
                    return None
                order_type = str(order.get("type", "")).lower()
                if order_type != "limit":
                    return None
                limit_price = order.get("limit_price")
                fill_price = order.get("filled_avg_price")
                if not limit_price or not fill_price:
                    return None
                try:
                    limit_price = float(limit_price)
                    fill_price = float(fill_price)
                    if limit_price <= 0:
                        return None
                    side = str(order.get("side", "")).lower()
                    # 正数=更差成交，负数=更优成交
                    if side == "buy":
                        return (fill_price - limit_price) / limit_price * 100
                    return (limit_price - fill_price) / limit_price * 100
                except Exception:
                    return None

            fill_minutes = [m for m in (_fill_minutes(o) for o in filled_orders) if m is not None]
            slip_values = [s for s in (_slippage_pct(o) for o in filled_orders) if s is not None]

            m1, m2, m3 = st.columns(3)
            m1.metric("已成交笔数", len(filled_orders))
            m2.metric("成交率", f"{(len(filled_orders) / len(closed_orders) * 100):.1f}%")
            m3.metric("平均成交耗时", f"{(sum(fill_minutes) / len(fill_minutes)):.1f} 分钟" if fill_minutes else "-")

            st.caption("限价滑点: 正数=吃亏，负数=优于限价")
            m4, m5 = st.columns(2)
            m4.metric("平均限价滑点", f"{(sum(slip_values) / len(slip_values)):+.2f}%" if slip_values else "-")
            m5.metric("最差限价滑点", f"{max(slip_values):+.2f}%" if slip_values else "-")

            detail_rows = []
            for o in closed_orders[:15]:
                mins = _fill_minutes(o)
                slip = _slippage_pct(o)
                detail_rows.append({
                    "订单ID": str(o.get("id", ""))[:8] + "...",
                    "股票": o.get("symbol", ""),
                    "方向": "买入" if o.get("side") == "buy" else "卖出",
                    "类型": o.get("type", ""),
                    "状态": o.get("status", ""),
                    "成交耗时": f"{mins:.1f}m" if mins is not None else "-",
                    "限价滑点": f"{slip:+.2f}%" if slip is not None else "-"
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)




def render_ai_center_page():
    """🤖 AI中心 - 重新设计: 智能选股 + 模型管理 + 博主追踪"""
    st.header("🤖 AI 选股中心")
    
    tab1, tab2, tab3 = st.tabs(["🎯 今日精选", "⚙️ 模型管理", "📢 博主追踪"])
    
    with tab1:
        render_ai_smart_picks()
    
    with tab2:
        render_ml_prediction_page()  # 保留原有模型管理
    
    with tab3:
        render_blogger_page()


def render_ai_smart_picks():
    """🎯 AI智能选股 - 核心推荐页面"""
    from pathlib import Path
    
    st.markdown("""
    <style>
    .pick-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        border-left: 4px solid #00C853;
    }
    .pick-card.warning {
        border-left-color: #FFD600;
    }
    .star-rating {
        color: #FFD700;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 选项
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="ai_pick_market")
    with col2:
        horizon_options = {"短线 (1-5天)": "short", "中线 (10-30天)": "medium", "长线 (60天+)": "long"}
        horizon_label = st.selectbox("交易周期", list(horizon_options.keys()), key="ai_horizon")
        horizon = horizon_options[horizon_label]
    with col3:
        max_picks = st.selectbox("推荐数量", [3, 5, 8, 10], index=1, key="ai_max_picks")
    
    # 检查模型状态
    model_dir = Path(__file__).parent / "ml" / "saved_models" / f"v2_{market.lower()}"
    return_model_exists = (model_dir / "return_5d.joblib").exists()
    ranker_model_exists = (model_dir / f"ranker_{horizon}.joblib").exists()
    
    # 模型状态显示
    status_cols = st.columns(2)
    with status_cols[0]:
        if return_model_exists:
            st.success("✓ 收益预测模型已加载", icon="🎯")
        else:
            st.warning("⚠ 收益预测模型未训练", icon="⚠️")
    with status_cols[1]:
        if ranker_model_exists:
            st.success(f"✓ 排序模型 ({horizon}) 已加载", icon="🏆")
        else:
            st.info(f"💡 排序模型 ({horizon}) 未训练，使用规则引擎")
    
    st.divider()
    
    # 获取推荐
    if st.button("🔄 刷新推荐", type="primary", key="refresh_ai_picks"):
        st.session_state['ai_picks_loaded'] = False
    
    # 加载推荐
    with st.spinner("AI 分析中..."):
        try:
            from ml.smart_picker import get_todays_picks, SmartPicker
            from db.database import get_connection
            from db.stock_history import get_stock_history
            
            # 获取最新信号
            conn = get_connection()
            query = """
                SELECT DISTINCT symbol, scan_date, price, 
                       COALESCE(blue_daily, 0) as blue_daily,
                       COALESCE(blue_weekly, 0) as blue_weekly,
                       COALESCE(blue_monthly, 0) as blue_monthly,
                       COALESCE(is_heima, 0) as is_heima,
                       company_name
                FROM scan_results
                WHERE market = ?
                ORDER BY scan_date DESC
                LIMIT 100
            """
            signals_df = pd.read_sql_query(query, conn, params=(market,))
            conn.close()
            
            if signals_df.empty:
                st.warning("暂无信号数据，请先运行扫描")
                return
            
            latest_date = signals_df['scan_date'].iloc[0]
            today_signals = signals_df[signals_df['scan_date'] == latest_date]
            
            st.caption(f"📅 信号日期: {latest_date} | 共 {len(today_signals)} 只股票")
            
            # 获取价格历史
            price_history = {}
            progress = st.progress(0)
            symbols = today_signals['symbol'].unique()
            
            for i, symbol in enumerate(symbols):
                history = get_stock_history(symbol, market, days=100)
                if not history.empty:
                    price_history[symbol] = history
                progress.progress((i + 1) / len(symbols))
            progress.empty()
            
            # 智能选股 (使用排序模型)
            picker = SmartPicker(market=market, horizon=horizon)
            picks = picker.pick(today_signals, price_history, max_picks=max_picks)
            
            # 📊 自动记录 ML 预测到追踪表
            try:
                from services.ml_prediction_tracker import log_predictions_batch
                pick_dicts = [p.to_dict() for p in picks]
                logged = log_predictions_batch(pick_dicts, market, latest_date)
                if logged > 0:
                    logger.info(f"Logged {logged} ML predictions for {latest_date}")
            except Exception as e:
                logger.warning(f"ML prediction logging failed (non-fatal): {e}")
            
            if not picks:
                st.info("今日没有高置信度的推荐")
                return
            
            # === 显示推荐 ===
            st.markdown(f"### 🎯 今日精选 ({len(picks)} 只)")
            
            # 汇总统计
            avg_score = sum(p.overall_score for p in picks) / len(picks)
            avg_rr = sum(p.risk_reward_ratio for p in picks) / len(picks)
            high_conf = sum(1 for p in picks if p.star_rating >= 4)
            
            sum_cols = st.columns(4)
            with sum_cols[0]:
                st.metric("平均评分", f"{avg_score:.0f}/100")
            with sum_cols[1]:
                st.metric("高置信度", f"{high_conf}/{len(picks)}")
            with sum_cols[2]:
                st.metric("平均风险收益比", f"1:{avg_rr:.1f}")
            with sum_cols[3]:
                avg_pred = sum(p.pred_return_5d for p in picks) / len(picks)
                st.metric("平均预测收益", f"{avg_pred:+.1f}%")
            
            st.divider()
            
            # 详细推荐卡片
            for i, pick in enumerate(picks):
                stars = "⭐" * pick.star_rating + "☆" * (5 - pick.star_rating)
                
                # 卡片颜色
                if pick.star_rating >= 4:
                    card_border = "#00C853"
                    card_bg = "#1a472a"
                elif pick.star_rating >= 3:
                    card_border = "#FFD600"
                    card_bg = "#4a4a00"
                else:
                    card_border = "#666"
                    card_bg = "#333"
                
                # 价格符号
                price_sym = "¥" if market == "CN" else "$"
                
                with st.container():
                    # 头部: 股票名称 + 评分
                    header_col1, header_col2 = st.columns([3, 1])
                    with header_col1:
                        display_name = pick.name if pick.name else pick.symbol
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 1.5em; font-weight: bold;">{display_name}</span>
                            <span style="color: #888; font-size: 0.9em;">{pick.symbol}</span>
                            <span class="star-rating">{stars}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with header_col2:
                        st.markdown(f"""
                        <div style="text-align: right;">
                            <span style="font-size: 1.3em; font-weight: bold;">{price_sym}{pick.price:.2f}</span>
                            <br>
                            <span style="font-size: 1.1em; color: {'#00C853' if pick.pred_return_5d > 0 else '#FF5252'};">
                                {pick.pred_return_5d:+.1f}% 预测
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 内容区
                    content_cols = st.columns([1, 1, 1])
                    
                    with content_cols[0]:
                        st.markdown("**📊 信号验证**")
                        for signal in pick.signals_confirmed[:4]:
                            st.markdown(f"<span style='color: #00C853;'>{signal}</span>", unsafe_allow_html=True)
                        for warning in pick.signals_warning[:2]:
                            st.markdown(f"<span style='color: #FFD600;'>{warning}</span>", unsafe_allow_html=True)
                    
                    with content_cols[1]:
                        st.markdown("**🎯 交易计划**")
                        st.markdown(f"""
                        - 止损: {price_sym}{pick.stop_loss_price:.2f} ({pick.stop_loss_pct:+.1f}%)
                        - 目标: {price_sym}{pick.target_price:.2f} (+{pick.target_pct:.1f}%)
                        - 风险收益比: **1:{pick.risk_reward_ratio:.1f}**
                        """)
                    
                    with content_cols[2]:
                        st.markdown("**💡 AI 预测**")
                        # 获取当前周期的排名分
                        rank_score = pick.rank_score_short
                        if horizon == 'medium':
                            rank_score = pick.rank_score_medium
                        elif horizon == 'long':
                            rank_score = pick.rank_score_long
                        
                        pred_20d = getattr(pick, 'pred_return_20d', 0) or 0
                        pred_dd = getattr(pick, 'pred_max_dd', 0) or 0
                        
                        st.markdown(f"""
                        - 5日预测: **{pick.pred_return_5d:+.1f}%**
                        - 20日预测: **{pred_20d:+.1f}%**
                        - 上涨概率: **{pick.pred_direction_prob:.0%}**
                        - 预测回撤: **{pred_dd:+.1f}%**
                        - 仓位建议: **{pick.suggested_position_pct:.0f}%**
                        - 综合评分: **{pick.overall_score:.0f}**/100
                        """)
                    
                    # 指标徽章
                    dd_badge = ""
                    pred_dd = getattr(pick, 'pred_max_dd', 0) or 0
                    if pred_dd < -5:
                        dd_badge = f"""
                        <span style="background: #FF525233; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">
                            ⚠️ 回撤 {pred_dd:+.0f}%
                        </span>"""
                    
                    dir_color = "#00C853" if pick.pred_direction_prob > 0.6 else "#FFD600" if pick.pred_direction_prob > 0.5 else "#FF5252"
                    
                    st.markdown(f"""
                    <div style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
                        <span style="background: {dir_color}33; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">
                            🎯 方向 {pick.pred_direction_prob:.0%}
                        </span>
                        <span style="background: #E91E6333; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">
                            🏆 排名分 {rank_score:.0f}
                        </span>
                        <span style="background: #00C85333; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            日B {pick.blue_daily:.0f}
                        </span>
                        <span style="background: #FFD60033; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            周B {pick.blue_weekly:.0f}
                        </span>
                        <span style="background: #2196F333; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            月B {pick.blue_monthly:.0f}
                        </span>
                        <span style="background: #9C27B033; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            RSI {pick.rsi:.0f}
                        </span>
                        <span style="background: #FF572233; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            量比 {pick.volume_ratio:.1f}x
                        </span>{dd_badge}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 操作按钮
                    btn_cols = st.columns([1, 1, 1, 3])
                    with btn_cols[0]:
                        if st.button("📈 查看K线", key=f"ai_chart_{pick.symbol}"):
                            st.session_state[f'ai_detail_{pick.symbol}'] = True
                    with btn_cols[1]:
                        if st.button("💰 模拟买入", key=f"ai_buy_{pick.symbol}"):
                            st.session_state[f'ai_buy_form_{pick.symbol}'] = True
                    with btn_cols[2]:
                        if st.button("👁️ 加入观察", key=f"ai_watch_{pick.symbol}"):
                            try:
                                from services.signal_tracker import add_to_watchlist
                                add_to_watchlist(
                                    pick.symbol, market,
                                    entry_price=pick.price,
                                    target_price=pick.target_price,
                                    stop_loss=pick.stop_loss_price
                                )
                                st.success(f"已加入观察列表")
                            except Exception as e:
                                st.error(f"添加失败: {e}")
                    
                    # 详情展开
                    if st.session_state.get(f'ai_detail_{pick.symbol}'):
                        with st.expander("📊 详细分析", expanded=True):
                            from components.stock_detail import render_unified_stock_detail
                            render_unified_stock_detail(
                                symbol=pick.symbol,
                                market=market,
                                key_prefix=f"ai_detail_{pick.symbol}"
                            )
                    
                    # 买入表单
                    if st.session_state.get(f'ai_buy_form_{pick.symbol}'):
                        with st.expander("💰 模拟买入", expanded=True):
                            buy_col1, buy_col2 = st.columns(2)
                            with buy_col1:
                                buy_shares = st.number_input(
                                    "买入数量", 
                                    min_value=1, 
                                    value=100,
                                    key=f"ai_buy_shares_{pick.symbol}"
                                )
                            with buy_col2:
                                buy_price = st.number_input(
                                    "买入价格",
                                    value=pick.price,
                                    key=f"ai_buy_price_{pick.symbol}"
                                )
                            
                            total_cost = buy_shares * buy_price
                            st.info(f"总成本: {price_sym}{total_cost:,.2f}")
                            
                            if st.button("确认买入", key=f"ai_confirm_buy_{pick.symbol}", type="primary"):
                                try:
                                    from services.portfolio_service import paper_buy
                                    result = paper_buy(
                                        symbol=pick.symbol,
                                        market=market,
                                        shares=buy_shares,
                                        price=buy_price
                                    )
                                    if result.get('success'):
                                        st.success(f"✅ 成功买入 {buy_shares} 股 {pick.symbol}")
                                        st.session_state[f'ai_buy_form_{pick.symbol}'] = False
                                    else:
                                        st.error(result.get('error', '买入失败'))
                                except Exception as e:
                                    st.error(f"买入失败: {e}")
                    
                    st.divider()
            
            # === 风险提示 ===
            st.markdown("""
            ---
            ### ⚠️ 风险提示
            
            - 以上推荐基于 **技术分析 + ML模型**，仅供参考
            - **严格执行止损**，保护本金是第一位的
            - 建议单只股票仓位不超过 **15%**
            - 历史表现不代表未来收益
            """)
            
        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def render_ml_prediction_page():
    """🎯 ML 模型预测页面 - 完整版"""
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path
    import json
    
    st.subheader("🎯 ML 智能选股")
    
    # 市场选择
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="ml_market")
    
    # 检查模型是否存在
    model_dir = Path(__file__).parent / "ml" / "saved_models" / f"v2_{market.lower()}"
    meta_path = model_dir / "return_predictor_meta.json"
    ranker_meta_path = model_dir / "ranker_meta.json"
    cost_profile_path = model_dir / "training_cost_profile.json"
    objective_path = model_dir / "training_objective.json"
    stability_path = model_dir / "feature_stability_report.json"
    walk_forward_path = model_dir / "walk_forward_report.json"
    
    if not meta_path.exists():
        st.warning("⚠️ 模型未训练")
        st.info("""
        **训练步骤:**
        ```bash
        cd versions/v3
        python ml/pipeline.py --market US --days 60
        ```
        """)
        
        if st.button("🚀 开始训练", key="train_full"):
            with st.spinner("训练中... (约 30 秒)"):
                try:
                    from ml.pipeline import train_pipeline
                    result = train_pipeline(market=market, days_back=60)
                    if result and result.get('status') == 'success':
                        st.success("✅ 训练完成!")
                        st.rerun()
                    else:
                        st.error("训练失败")
                except Exception as e:
                    st.error(f"训练失败: {e}")
        return
    
    # 加载模型元数据
    with open(meta_path) as f:
        meta = json.load(f)

    available_horizons = meta.get('horizons') or list((meta.get('metrics') or {}).keys())
    if not available_horizons:
        available_horizons = ["5d"]
    default_h = "20d" if "20d" in available_horizons else available_horizons[0]
    with col2:
        horizon = st.selectbox("预测周期", available_horizons, index=available_horizons.index(default_h), key="ml_horizon")
    
    # 加载排序模型元数据
    ranker_meta = {}
    if ranker_meta_path.exists():
        with open(ranker_meta_path) as f:
            ranker_meta = json.load(f)

    # 加载训练成本画像（毛/净收益对比）
    cost_profile = {}
    if cost_profile_path.exists():
        try:
            with open(cost_profile_path) as f:
                cost_profile = json.load(f)
        except Exception:
            cost_profile = {}

    objective_meta = {}
    if objective_path.exists():
        try:
            with open(objective_path) as f:
                objective_meta = json.load(f)
        except Exception:
            objective_meta = {}

    stability_meta = {}
    if stability_path.exists():
        try:
            with open(stability_path) as f:
                stability_meta = json.load(f)
        except Exception:
            stability_meta = {}

    walk_forward_meta = {}
    if walk_forward_path.exists():
        try:
            with open(walk_forward_path) as f:
                walk_forward_meta = json.load(f)
        except Exception:
            walk_forward_meta = {}

    # ==================================
    # 🛡️ 风控参数（全局）
    # ==================================
    try:
        from risk.trading_profile import load_trading_profile, save_trading_profile
        risk_cfg = load_trading_profile()
    except Exception:
        risk_cfg = {}
        save_trading_profile = None

    with st.expander("🛡️ 风控参数", expanded=False):
        st.caption("这些参数会影响 Smart Picker 的止损/止盈/仓位建议")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            atr_mult = st.slider("ATR止损倍数", 1.0, 4.0, float(risk_cfg.get("atr_stop_multiplier", 2.0)), 0.1, key="risk_atr_mult")
            max_stop = st.slider("最大止损(%)", 3.0, 15.0, float(risk_cfg.get("max_stop_loss_pct", 8.0)), 0.5, key="risk_max_stop")
            target_cap = st.slider("目标收益上限(%)", 5.0, 30.0, float(risk_cfg.get("target_cap_pct", 15.0)), 0.5, key="risk_target_cap")
        with rc2:
            rr_high = st.slider("高仓位RR阈值", 1.2, 3.0, float(risk_cfg.get("rr_high", 2.0)), 0.1, key="risk_rr_high")
            rr_mid = st.slider("中仓位RR阈值", 1.0, 2.5, float(risk_cfg.get("rr_mid", 1.5)), 0.1, key="risk_rr_mid")
            boost = st.slider("强信号目标放大", 1.0, 1.8, float(risk_cfg.get("strong_signal_target_boost", 1.2)), 0.05, key="risk_boost")
        with rc3:
            prob_high = st.slider("高仓位胜率阈值", 0.50, 0.70, float(risk_cfg.get("prob_high", 0.55)), 0.01, key="risk_prob_high")
            prob_mid = st.slider("中仓位胜率阈值", 0.45, 0.65, float(risk_cfg.get("prob_mid", 0.52)), 0.01, key="risk_prob_mid")
            pos_low = st.slider("保守仓位(%)", 1.0, 10.0, float(risk_cfg.get("position_low_pct", 5.0)), 0.5, key="risk_pos_low")

        rc4, rc5 = st.columns(2)
        with rc4:
            pos_mid = st.slider("中仓位(%)", 5.0, 20.0, float(risk_cfg.get("position_mid_pct", 10.0)), 0.5, key="risk_pos_mid")
        with rc5:
            pos_high = st.slider("高仓位(%)", 8.0, 30.0, float(risk_cfg.get("position_high_pct", 15.0)), 0.5, key="risk_pos_high")

        if st.button("💾 保存风控参数", key="save_risk_profile"):
            if save_trading_profile is None:
                st.error("风控配置模块不可用")
            else:
                payload = {
                    "atr_stop_multiplier": atr_mult,
                    "max_stop_loss_pct": max_stop,
                    "target_cap_pct": target_cap,
                    "strong_signal_target_boost": boost,
                    "rr_high": rr_high,
                    "rr_mid": rr_mid,
                    "prob_high": prob_high,
                    "prob_mid": prob_mid,
                    "position_high_pct": pos_high,
                    "position_mid_pct": pos_mid,
                    "position_low_pct": pos_low,
                }
                ok = save_trading_profile(payload)
                if ok:
                    st.success("✅ 风控参数已保存，新的智能选股会自动生效")
                else:
                    st.error("保存失败")
    
    # ==================================
    # 📊 模型概览 - 详细指标
    # ==================================
    st.markdown("### 📊 模型概览")
    
    model_tab1, model_tab2, model_tab3, model_tab4, model_tab5, model_tab6, model_tab7 = st.tabs([
        "📈 收益预测模型", "🏆 排序模型", "🔧 特征重要性", "⚙️ 超参数调优", "🔗 模型对比", "🧪 稳定性", "⚖️ 交易口径评估"
    ])
    
    with model_tab1:
        st.markdown("**Return Predictor** - 预测 5/20/60 天收益率（中长线优先）")

        if cost_profile:
            st.caption(
                f"训练标签已扣成本: 手续费 {cost_profile.get('commission_bps', 0):.1f}bps + "
                f"滑点 {cost_profile.get('slippage_bps', 0):.1f}bps（单边），"
                f"双边合计 {cost_profile.get('round_trip_cost_pct', 0):.2f}%"
            )
        if objective_meta:
            primary = ",".join(objective_meta.get("primary_horizons", []))
            st.caption(f"训练目标: {primary} 超额收益 + 回撤惩罚")

            horizon_cost_rows = []
            for h, v in (cost_profile.get('horizons') or {}).items():
                horizon_cost_rows.append({
                    '周期': h,
                    '毛收益': f"{v.get('avg_gross_return_pct', 0):+.2f}%",
                    '净收益': f"{v.get('avg_net_return_pct', 0):+.2f}%",
                    '成本拖累': f"{v.get('cost_drag_pct', 0):.2f}%",
                    '毛胜率': f"{v.get('gross_win_rate_pct', 0):.1f}%",
                    '净胜率': f"{v.get('net_win_rate_pct', 0):.1f}%",
                    '样本': int(v.get('samples', 0)),
                })
            if horizon_cost_rows:
                horizon_cost_df = pd.DataFrame(horizon_cost_rows)
                horizon_cost_df = horizon_cost_df[['周期', '毛收益', '净收益', '成本拖累', '毛胜率', '净胜率', '样本']]
                st.dataframe(horizon_cost_df, hide_index=True, use_container_width=True)
                st.caption("毛/净对比用于衡量策略交易成本敏感度，净收益更接近真实可交易表现。")
        
        # 所有周期指标对比表
        metrics_data = []
        for h, m in meta.get('metrics', {}).items():
            metrics_data.append({
                '周期': h,
                'R²': f"{m.get('r2', 0):.3f}",
                '方向准确率': f"{m.get('direction_accuracy', 0):.1%}",
                'RMSE': f"{m.get('rmse', 0):.2f}%",
                'MAE': f"{m.get('mae', 0):.2f}%",
                '训练样本': m.get('train_samples', 0),
                '测试样本': m.get('test_samples', 0)
            })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            # 缩短列名
            metrics_df.columns = ['周期', 'R²', '方向准确率', 'RMSE', 'MAE', '训练', '测试']
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            # 方向准确率图
            fig_acc = go.Figure()
            horizons = [m['周期'] for m in metrics_data]
            accuracies = [float(m['方向准确率'].replace('%', '')) for m in metrics_data]
            
            fig_acc.add_trace(go.Bar(
                x=horizons, y=accuracies,
                marker_color=['#2ecc71' if a > 60 else '#f39c12' if a > 50 else '#e74c3c' for a in accuracies],
                text=[f"{a:.1f}%" for a in accuracies],
                textposition='outside'
            ))
            fig_acc.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="随机基准 50%")
            fig_acc.update_layout(
                title="各周期方向准确率",
                xaxis_title="预测周期", yaxis_title="准确率 (%)",
                height=300, yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # 模型解读
        horizon_meta = meta.get('metrics', {}).get(horizon, {})
        if horizon_meta:
            r2 = horizon_meta.get('r2', 0)
            dir_acc = horizon_meta.get('direction_accuracy', 0)
            
            st.markdown(f"""
            **当前选择: {horizon}**
            - R² = {r2:.3f}: {"优秀" if r2 > 0.5 else "良好" if r2 > 0.3 else "一般" if r2 > 0.1 else "较弱"} (解释了 {r2*100:.1f}% 的收益变化)
            - 方向准确率 = {dir_acc:.1%}: {"优秀" if dir_acc > 0.7 else "良好" if dir_acc > 0.6 else "一般" if dir_acc > 0.55 else "较弱"}
            """)
    
    with model_tab2:
        st.markdown("**Signal Ranker** - 排序最可能赚钱的股票 (短/中/长线)")
        
        if ranker_meta.get('metrics'):
            ranker_data = []
            horizon_labels = {'short': '短线 (5天)', 'medium': '中线 (20天)', 'long': '长线 (60天)'}
            
            for h, m in ranker_meta.get('metrics', {}).items():
                group_display = m.get('n_groups')
                if group_display is None:
                    n_train_g = m.get('n_train_groups', 0)
                    n_test_g = m.get('n_test_groups', 0)
                    group_display = f"{n_train_g}/{n_test_g}"
                ranker_data.append({
                    '周期': horizon_labels.get(h, h),
                    'NDCG@10': f"{m.get('ndcg@10', 0):.3f}",
                    'Top10平均收益': f"{m.get('top10_avg_return', 0):+.2f}%",
                    '训练样本': m.get('train_samples', 0),
                    '分组数': group_display
                })
            
            ranker_df = pd.DataFrame(ranker_data)
            ranker_df.columns = ['周期', 'NDCG', 'Top10收益', '样本', '分组']
            st.dataframe(ranker_df, hide_index=True, use_container_width=True)
            
            st.markdown("""
            **指标说明:**
            - **NDCG@10**: 归一化折损累积增益，越接近 1 排序质量越好
            - **Top10平均收益**: 排名前 10 的股票平均实际收益
            """)
        else:
            st.info("排序模型未训练")

    with model_tab6:
        st.markdown("**稳健性验证** - 特征稳定性 + Walk-forward")

        if stability_meta:
            summary = stability_meta.get('summary', {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("特征数", int(summary.get('feature_count', 0)))
            with c2:
                st.metric("平均缺失率", f"{summary.get('avg_missing_rate', 0)*100:.1f}%")
            with c3:
                st.metric("平均漂移分", f"{summary.get('avg_drift_score', 0):.2f}")
            with c4:
                st.metric("平均|IC20|", f"{summary.get('avg_abs_ic20', 0):.3f}")

            stable_df = pd.DataFrame(stability_meta.get('top_stable_features', [])[:15])
            unstable_df = pd.DataFrame(stability_meta.get('top_unstable_features', [])[:15])
            lcol, rcol = st.columns(2)
            with lcol:
                st.caption("Top 稳定特征")
                if not stable_df.empty:
                    show = stable_df[['feature', 'stability_score', 'missing_rate', 'drift_score', 'ic_20d', 'ic_60d']]
                    st.dataframe(show, hide_index=True, use_container_width=True)
            with rcol:
                st.caption("Top 不稳定特征")
                if not unstable_df.empty:
                    show = unstable_df[['feature', 'stability_score', 'missing_rate', 'drift_score', 'ic_20d', 'ic_60d']]
                    st.dataframe(show, hide_index=True, use_container_width=True)
        else:
            st.info("暂无特征稳定性报告，请先训练一次模型。")

        st.divider()
        if walk_forward_meta:
            status = walk_forward_meta.get('status')
            if status == 'ok':
                w1, w2, w3 = st.columns(3)
                with w1:
                    st.metric("Walk-forward 折数", int(walk_forward_meta.get('n_folds', 0)))
                with w2:
                    st.metric("平均Spearman IC", f"{walk_forward_meta.get('avg_spearman_ic', 0):.3f}")
                with w3:
                    st.metric("Top20平均收益", f"{walk_forward_meta.get('avg_top20_return', 0):+.2f}%")
            else:
                st.warning(f"Walk-forward 未运行: {walk_forward_meta.get('reason', 'unknown')}")
        else:
            st.info("暂无 Walk-forward 报告，请先训练一次模型。")
    
    with model_tab3:
        st.markdown("**特征重要性** - 哪些特征对预测最重要")
        
        try:
            import joblib
            model_path = model_dir / f"return_{horizon}.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                feature_names = meta.get('feature_names', [])
                
                if hasattr(model, 'feature_importances_') and feature_names:
                    importance = dict(zip(feature_names, model.feature_importances_))
                    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Top 20 特征
                    top20 = sorted_imp[:20]
                    
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(
                        y=[f[0] for f in top20][::-1],
                        x=[f[1] for f in top20][::-1],
                        orientation='h',
                        marker_color='steelblue'
                    ))
                    fig_imp.update_layout(
                        title=f"Top 20 重要特征 ({horizon})",
                        xaxis_title="重要性得分",
                        height=500,
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # 特征分类统计
                    categories = {
                        '均线特征': [f for f in feature_names if 'ma_' in f or 'ema_' in f],
                        '动量特征': [f for f in feature_names if 'momentum' in f or 'roc' in f or 'return' in f],
                        '波动率特征': [f for f in feature_names if 'volatility' in f or 'atr' in f],
                        'RSI特征': [f for f in feature_names if 'rsi' in f],
                        'MACD特征': [f for f in feature_names if 'macd' in f],
                        'KDJ特征': [f for f in feature_names if 'kdj' in f],
                        '布林带特征': [f for f in feature_names if 'bb_' in f],
                        '成交量特征': [f for f in feature_names if 'volume' in f or 'obv' in f],
                        'K线形态': [f for f in feature_names if 'body' in f or 'shadow' in f or 'doji' in f or 'hammer' in f],
                        'BLUE信号': [f for f in feature_names if 'blue' in f],
                    }
                    
                    cat_importance = []
                    for cat, feats in categories.items():
                        total_imp = sum(importance.get(f, 0) for f in feats)
                        cat_importance.append({'类别': cat, '总重要性': total_imp, '特征数': len(feats)})
                    
                    cat_df = pd.DataFrame(cat_importance).sort_values('总重要性', ascending=False)
                    cat_df['总重要性'] = cat_df['总重要性'].apply(lambda x: f"{x:.4f}")
                    cat_df.columns = ['类别', '重要性', '特征数']
                    
                    st.markdown("**特征类别重要性汇总:**")
                    st.dataframe(cat_df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.warning(f"无法加载特征重要性: {e}")
    
    with model_tab4:
        st.markdown("**Hyperparameter Tuning** - GridSearch 找最优参数")
        
        # 检查是否有调优结果
        tuning_path = model_dir.parent.parent / 'tuning_results' / market.lower() / 'best_params.json'
        
        if tuning_path.exists():
            with open(tuning_path) as f:
                best_params = json.load(f)
            
            st.success("✅ 已有调优结果")
            
            # 显示最优参数
            for model_key, params in best_params.items():
                with st.expander(f"📊 {model_key}", expanded=True):
                    params_df = pd.DataFrame([
                        {'参数': k, '最优值': v} for k, v in params.items()
                    ])
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
            
            # 加载调优历史
            history_path = tuning_path.parent / 'tuning_history.json'
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                
                if history:
                    st.markdown("**调优效果对比:**")
                    history_df = pd.DataFrame(history)
                    history_df['提升'] = history_df['improvement'].apply(lambda x: f"{x:+.1f}%")
                    history_df['最优分数'] = history_df['best_score'].apply(lambda x: f"{x:.3f}")
                    history_df['默认分数'] = history_df['default_score'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(
                        history_df[['horizon', '默认分数', '最优分数', '提升']].rename(
                            columns={'horizon': '周期'}
                        ),
                        hide_index=True, use_container_width=True
                    )
        else:
            st.info("暂无调优结果")
        
        st.markdown("---")
        
        # 调优按钮
        col1, col2 = st.columns(2)
        with col1:
            fast_mode = st.checkbox("快速模式", value=True, help="使用较小的搜索空间")
        with col2:
            n_iter = st.slider("搜索次数", 10, 100, 30, help="RandomizedSearch 迭代次数")
        
        if st.button("🔧 开始调优", key="start_tuning", type="primary"):
            with st.spinner("调优中... (可能需要几分钟)"):
                try:
                    from ml.hyperparameter_tuning import run_tuning
                    results = run_tuning(market=market, fast=fast_mode)
                    
                    if results:
                        st.success("✅ 调优完成!")
                        st.rerun()
                    else:
                        st.error("调优失败")
                except Exception as e:
                    st.error(f"调优出错: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("""
        **说明:**
        - 快速模式: 约 2-5 分钟
        - 完整模式: 约 10-30 分钟
        - 调优使用 5 折交叉验证
        - 优化目标: 方向准确率
        """)
    
    with model_tab5:
        st.markdown("**Model Comparison** - 独立模型 vs 串联模型")
        
        st.markdown("""
        **两种架构:**
        
        | 模式 | 架构 | 特点 |
        |------|------|------|
        | 独立模型 | ReturnPredictor + SignalRanker 各自独立 | 简单，训练快 |
        | 串联模型 | ReturnPredictor → 预测特征 → SignalRanker | Ranker可学习"哪些预测更可信" |
        """)
        
        # 检查是否有对比结果
        comparison_path = model_dir / 'model_comparison.json'
        
        if comparison_path.exists():
            with open(comparison_path) as f:
                comparison = json.load(f)
            
            st.success("✅ 已有对比结果")
            
            # 显示对比表
            if 'comparison' in comparison:
                comp_df = pd.DataFrame(comparison['comparison'])
                comp_df['independent_ndcg'] = comp_df['independent_ndcg'].apply(lambda x: f"{x:.3f}")
                comp_df['ensemble_ndcg'] = comp_df['ensemble_ndcg'].apply(lambda x: f"{x:.3f}")
                comp_df['improvement'] = comp_df['improvement'].apply(lambda x: f"{x:+.1f}%")
                comp_df.columns = ['周期', '独立模型 NDCG', '串联模型 NDCG', '提升']
                
                st.markdown("**排序模型 NDCG@10 对比:**")
                st.dataframe(comp_df, hide_index=True, use_container_width=True)
                
                # 添加特征信息
                if 'ensemble' in comparison:
                    added = comparison['ensemble'].get('added_features', [])
                    if added:
                        st.markdown(f"**串联模型新增特征:** `{', '.join(added)}`")
        else:
            st.info("暂无对比结果")
        
        st.markdown("---")
        
        if st.button("🔗 运行模型对比", key="run_comparison", type="primary"):
            with st.spinner("训练并对比中... (约 1-2 分钟)"):
                try:
                    from ml.pipeline import MLPipeline
                    from ml.models.ensemble_predictor import compare_models
                    
                    # 准备数据
                    pipeline = MLPipeline(market=market)
                    X, returns_dict, drawdowns_dict, groups, feature_names, _ = pipeline.prepare_dataset()
                    
                    if X is not None and len(X) > 0:
                        # 运行对比
                        results = compare_models(X, returns_dict, drawdowns_dict, groups, feature_names)
                        
                        # 保存结果
                        with open(comparison_path, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        st.success("✅ 对比完成!")
                        st.rerun()
                    else:
                        st.error("无法准备数据")
                except Exception as e:
                    st.error(f"对比出错: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("""
        **串联模型新增特征:**
        - `pred_return_1d/5d/10d/30d`: 预测收益
        - `pred_return_mean`: 预测收益均值
        - `pred_return_std`: 预测不确定性
        - `pred_momentum`: 长短期预测差异
        - `pred_direction_consistency`: 方向一致性
        """)

    with model_tab7:
        st.markdown("**同一交易口径评估** - 基线策略 vs ML 排序模型")
        st.caption("在相同平仓规则下对比胜率/收益，避免“规则不一致”导致的误判。")

        c1, c2, c3 = st.columns(3)
        with c1:
            eval_days_back = st.slider("评估回溯天数", 15, 240, 90, 5, key=f"ml_eval_days_back_{market}")
        with c2:
            eval_topk = st.slider("每日入选数 TopK", 1, 20, 5, 1, key=f"ml_eval_topk_{market}")
        with c3:
            eval_max_rows = st.slider("最大样本数", 120, 1500, 600, 60, key=f"ml_eval_max_rows_{market}")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            eval_exit_rule = st.selectbox(
                "平仓规则",
                ["fixed_5d", "fixed_10d", "fixed_20d", "tp_sl_time", "kdj_dead_cross", "top_divergence_guard", "duokongwang_sell"],
                index=1,
                key=f"ml_eval_exit_rule_{market}",
            )
        with r2:
            eval_tp = st.slider("止盈(%)", 3.0, 30.0, 10.0, 0.5, key=f"ml_eval_tp_{market}")
        with r3:
            eval_sl = st.slider("止损(%)", 2.0, 20.0, 6.0, 0.5, key=f"ml_eval_sl_{market}")
        with r4:
            eval_hold = st.slider("最大持有天数", 3, 60, 20, 1, key=f"ml_eval_hold_{market}")

        if eval_exit_rule not in {"tp_sl_time", "top_divergence_guard"}:
            st.caption("提示: 当前规则不使用止盈/止损参数；仅 `tp_sl_time` 与 `top_divergence_guard` 使用这些参数。")

        if st.button("🚀 运行同口径评估", key=f"ml_vs_baseline_run_{market}", type="primary"):
            with st.spinner("评估中，正在按日重排并回放平仓规则..."):
                eval_ret = _evaluate_ml_vs_baseline(
                    market=market,
                    days_back=eval_days_back,
                    topk_per_day=eval_topk,
                    exit_rule=eval_exit_rule,
                    take_profit_pct=eval_tp,
                    stop_loss_pct=eval_sl,
                    max_hold_days=eval_hold,
                    max_eval_rows=eval_max_rows,
                )

            if not eval_ret.get("ok"):
                st.warning("暂无可评估样本。请先在“每日机会”里完成候选追踪快照。")
            else:
                base = eval_ret.get("base", {}) or {}
                ml = eval_ret.get("ml", {}) or {}
                meta_eval = eval_ret.get("meta", {}) or {}

                base_sample = int(base.get("sample") or 0)
                ml_sample = int(ml.get("sample") or 0)
                if base_sample == 0 or ml_sample == 0:
                    st.warning("样本不足，暂时无法形成有效对比。")

                base_wr = float(base.get("win_rate_pct") or 0.0)
                ml_wr = float(ml.get("win_rate_pct") or 0.0)
                base_ret = float(base.get("avg_return_pct") or 0.0)
                ml_ret = float(ml.get("avg_return_pct") or 0.0)
                base_exit = float(base.get("avg_exit_day") or 0.0)
                ml_exit = float(ml.get("avg_exit_day") or 0.0)

                d_wr = ml_wr - base_wr
                d_ret = ml_ret - base_ret
                d_exit = ml_exit - base_exit

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("基线胜率", f"{base_wr:.1f}%", help=f"样本 {base_sample}")
                with m2:
                    st.metric("ML胜率", f"{ml_wr:.1f}%", f"{d_wr:+.1f}%", help=f"样本 {ml_sample}")
                with m3:
                    st.metric("基线平均收益", f"{base_ret:+.2f}%")
                with m4:
                    st.metric("ML平均收益", f"{ml_ret:+.2f}%", f"{d_ret:+.2f}%")

                n1, n2, n3 = st.columns(3)
                with n1:
                    st.metric("基线平均持有", f"{base_exit:.1f} 天")
                with n2:
                    st.metric("ML平均持有", f"{ml_exit:.1f} 天", f"{d_exit:+.1f} 天")
                with n3:
                    ranked_cnt = int(meta_eval.get("ranked_cnt") or 0)
                    fallback_cnt = int(meta_eval.get("fallback_cnt") or 0)
                    fallback_ratio = (fallback_cnt / (ranked_cnt + fallback_cnt) * 100.0) if (ranked_cnt + fallback_cnt) > 0 else 0.0
                    st.metric("排序回退占比", f"{fallback_ratio:.1f}%", help="回退表示该样本未成功走到ML排序，使用BLUE加权替代")

                cmp_df = pd.DataFrame(
                    [
                        {
                            "策略": "基线 TopK(BLUE)",
                            "样本数": base_sample,
                            "胜率(%)": round(base_wr, 1),
                            "平均收益(%)": round(base_ret, 2),
                            "平均平仓天数": round(base_exit, 1),
                        },
                        {
                            "策略": "ML TopK(排序模型)",
                            "样本数": ml_sample,
                            "胜率(%)": round(ml_wr, 1),
                            "平均收益(%)": round(ml_ret, 2),
                            "平均平仓天数": round(ml_exit, 1),
                        },
                    ]
                )
                st.dataframe(cmp_df, hide_index=True, use_container_width=True)
                st.caption(
                    f"评估范围: 最近 {int(meta_eval.get('days_back') or 0)} 天 | "
                    f"覆盖交易日 {int(meta_eval.get('n_days') or 0)} | "
                    f"输入样本 {int(meta_eval.get('input_rows') or 0)}"
                )

    st.divider()
    
    # ==================================
    # 当前周期的核心指标卡片
    # ==================================
    horizon_meta = meta.get('metrics', {}).get(horizon, {})
    if horizon_meta:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            r2 = horizon_meta.get('r2', 0)
            st.metric("R²", f"{r2:.3f}", help="决定系数，越高模型解释力越强")
        with col2:
            dir_acc = horizon_meta.get('direction_accuracy', 0)
            delta = f"+{(dir_acc-0.5)*100:.0f}%" if dir_acc > 0.5 else f"{(dir_acc-0.5)*100:.0f}%"
            st.metric("方向准确率", f"{dir_acc:.1%}", delta=delta, help="预测涨跌方向的准确率")
        with col3:
            st.metric("RMSE", f"{horizon_meta.get('rmse', 0):.2f}%", help="均方根误差，越低越好")
        with col4:
            st.metric("样本数", f"{horizon_meta.get('train_samples', 0):,}", help="训练样本数量")
    
    st.divider()
    
    # === 加载今日信号 ===
    from db.database import get_connection
    from db.stock_history import get_stock_history
    from ml.features.feature_calculator import FeatureCalculator
    
    conn = get_connection()
    
    # 获取最新信号
    query = """
        SELECT DISTINCT symbol, scan_date, price, 
               COALESCE(blue_daily, 0) as blue_daily,
               COALESCE(blue_weekly, 0) as blue_weekly,
               COALESCE(blue_monthly, 0) as blue_monthly,
               COALESCE(is_heima, 0) as is_heima
        FROM scan_results
        WHERE market = ?
        ORDER BY scan_date DESC
        LIMIT 200
    """
    signals_df = pd.read_sql_query(query, conn, params=(market,))
    conn.close()
    
    if signals_df.empty:
        st.info("暂无信号数据")
        return
    
    latest_date = signals_df['scan_date'].iloc[0]
    today_signals = signals_df[signals_df['scan_date'] == latest_date]
    
    st.markdown(f"### 📈 {latest_date} 信号预测 ({len(today_signals)} 只)")
    
    # 加载模型
    try:
        import joblib
        return_model = joblib.load(model_dir / f"return_{horizon}.joblib")
        feature_names = json.load(open(model_dir / "feature_names.json"))
        
        # 为每个信号计算特征并预测
        calc = FeatureCalculator()
        predictions = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, (_, signal) in enumerate(today_signals.iterrows()):
            symbol = signal['symbol']
            
            # 获取历史数据
            history = get_stock_history(symbol, market, days=100)
            
            if history.empty or len(history) < 60:
                continue
            
            # 计算特征
            blue_signals = {
                'blue_daily': signal['blue_daily'],
                'blue_weekly': signal['blue_weekly'],
                'blue_monthly': signal['blue_monthly'],
                'is_heima': signal['is_heima'],
                'is_juedi': 0
            }
            
            features = calc.get_latest_features(history, blue_signals)
            
            if not features:
                continue
            
            # 准备特征向量
            X = np.array([[features.get(f, 0) for f in feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1e6, 1e6)
            
            # 预测
            pred_return = return_model.predict(X)[0]
            
            predictions.append({
                'symbol': symbol,
                'price': signal['price'],
                'blue_daily': signal['blue_daily'],
                'blue_weekly': signal['blue_weekly'],
                'is_heima': signal['is_heima'],
                f'pred_{horizon}': pred_return,
                'direction': '📈' if pred_return > 0 else '📉'
            })
            
            progress.progress((i + 1) / len(today_signals))
            status.text(f"处理: {symbol} ({i+1}/{len(today_signals)})")
        
        progress.empty()
        status.empty()
        
        if not predictions:
            st.warning("无法计算预测 (缺少历史数据)")
            return
        
        # 结果 DataFrame
        result_df = pd.DataFrame(predictions)
        result_df = result_df.sort_values(f'pred_{horizon}', ascending=False)
        result_df['rank'] = range(1, len(result_df) + 1)
        
        # === 显示 Top 10 ===
        st.markdown("### 🏆 Top 10 推荐")
        
        top10 = result_df.head(10).copy()
        top10['heima'] = top10['is_heima'].apply(lambda x: '⭐' if x else '')
        
        # 直接用 dataframe，列名简短
        show_cols = {
            'rank': '#',
            'symbol': '代码', 
            f'pred_{horizon}': '预测%',
            'direction': '↑↓',
            'blue_daily': '日B',
            'blue_weekly': '周B', 
            'heima': '🐴',
            'price': '$'
        }
        show_df = top10[list(show_cols.keys())].rename(columns=show_cols)
        show_df['预测%'] = show_df['预测%'].apply(lambda x: f"{x:+.1f}")
        show_df['$'] = show_df['$'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(show_df, hide_index=True, use_container_width=True)
        
        # === 预测分布 ===
        st.markdown("### 📊 预测分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 收益分布图
            fig = px.histogram(
                result_df, 
                x=f'pred_{horizon}',
                nbins=20,
                title=f"{horizon} 预测收益分布",
                labels={f'pred_{horizon}': '预测收益 (%)', 'count': '数量'}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 统计
            positive = (result_df[f'pred_{horizon}'] > 0).sum()
            negative = (result_df[f'pred_{horizon}'] <= 0).sum()
            avg_return = result_df[f'pred_{horizon}'].mean()
            
            st.metric("📈 预测上涨", f"{positive} 只")
            st.metric("📉 预测下跌", f"{negative} 只")
            st.metric("平均预测收益", f"{avg_return:+.1f}%")
        
        # === Bottom 10 ===
        with st.expander("📉 Bottom 10 (预测下跌最多)", expanded=False):
            bottom10 = result_df.tail(10).copy()
            bottom10 = bottom10.iloc[::-1]
            bottom10['heima'] = bottom10['is_heima'].apply(lambda x: '⭐' if x else '')
            
            show_cols = {
                'rank': '#',
                'symbol': '代码', 
                f'pred_{horizon}': '预测%',
                'direction': '↑↓',
                'blue_daily': '日B',
                'blue_weekly': '周B', 
                'heima': '🐴',
                'price': '$'
            }
            show_df2 = bottom10[list(show_cols.keys())].rename(columns=show_cols)
            show_df2['预测%'] = show_df2['预测%'].apply(lambda x: f"{x:+.1f}")
            show_df2['$'] = show_df2['$'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(show_df2, hide_index=True, use_container_width=True)
        
    except Exception as e:
        st.error(f"预测失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # ==================================
    # 📦 数据管理
    # ==================================
    st.divider()
    
    with st.expander("📦 数据管理", expanded=False):
        st.markdown("**历史K线数据** - 用于训练ML模型")
        
        # 数据统计
        try:
            from db.stock_history import get_history_stats
            from db.database import get_connection
            
            stats = get_history_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("K线股票数", stats.get('total_symbols', 0))
            with col2:
                st.metric("K线记录数", f"{stats.get('total_records', 0):,}")
            with col3:
                # 获取信号股票数
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(DISTINCT symbol) FROM scan_results WHERE market = ?', (market,))
                signal_count = cursor.fetchone()[0]
                conn.close()
                coverage = stats.get('total_symbols', 0) / signal_count * 100 if signal_count > 0 else 0
                st.metric("数据覆盖率", f"{coverage:.1f}%")
            
            # 缺失数据提示
            missing = signal_count - stats.get('total_symbols', 0)
            if missing > 0:
                st.warning(f"⚠️ 有 {missing} 只信号股票缺少历史数据")
        except Exception as e:
            st.warning(f"获取统计失败: {e}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            max_fetch = st.slider("获取数量", 50, 1000, 200, 50, help="一次获取多少只股票")
        with col2:
            fetch_days = st.slider("历史天数", 90, 730, 365, 30, help="获取多少天历史")
        
        if st.button("📥 获取更多数据", key="fetch_more_data"):
            with st.spinner(f"获取中... (约 {max_fetch * 0.5 / 60:.1f} 分钟)"):
                try:
                    from ml.batch_fetch_data import run_fetch
                    result = run_fetch(
                        market=market,
                        max_symbols=max_fetch,
                        days=fetch_days,
                        delay=0.3
                    )
                    
                    if result['success'] > 0:
                        st.success(f"✅ 获取完成! 成功: {result['success']}, 失败: {result['failed']}")
                        st.rerun()
                    else:
                        st.info("没有新数据需要获取")
                except Exception as e:
                    st.error(f"获取失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.caption("💡 数据来源: Polygon API (优先) / yfinance (备用)")


# --- V3 主导航 (精简版 4 入口) ---
def render_qlib_mining_hub():
    """主应用内的 Qlib 挖掘中心（不依赖 Streamlit 多页面发现）。"""
    import json
    import subprocess
    from pathlib import Path

    st.header("🧠 Qlib 因子与策略挖掘")
    st.caption("在主应用内查看/运行 Qlib 挖掘，不依赖 pages 侧栏菜单")

    try:
        from ml.qlib_integration import check_qlib_status
        status = check_qlib_status()
    except Exception:
        status = {"installed": False, "us_data": False, "cn_data": False}

    c1, c2, c3 = st.columns(3)
    c1.metric("Qlib 安装", "✅" if status.get("installed") else "❌")
    c2.metric("US 数据", "✅" if status.get("us_data") else "❌")
    c3.metric("CN 数据", "✅" if status.get("cn_data") else "❌")

    with st.expander("运行挖掘", expanded=False):
        market = st.selectbox("市场", ["US", "CN"], index=0, key="qlib_hub_market")
        segment = st.selectbox("市值分层", ["ALL", "LARGE", "MID", "SMALL"], index=0, key="qlib_hub_segment")
        days = st.slider("回溯天数", 180, 1460, 730, 30, key="qlib_hub_days")
        topk_grid = st.text_input("TopK 网格", "5,8,10,15", key="qlib_hub_topk")
        drop_grid = st.text_input("N_drop 网格", "1,2,3", key="qlib_hub_drop")
        run_batch = st.checkbox("批量跑分层对比（仅US）", value=True, key="qlib_hub_batch")

        if st.button("开始挖掘", type="primary", key="qlib_hub_run"):
            cmd = [
                sys.executable,
                "scripts/run_qlib_mining.py",
                "--market", market,
                "--segment", segment,
                "--days", str(days),
                "--topk-grid", topk_grid,
                "--drop-grid", drop_grid,
            ]
            if run_batch and market == "US":
                cmd.append("--run-segment-batch")

            with st.spinner("运行中，可能需要几分钟..."):
                proc = subprocess.run(
                    cmd,
                    cwd=str(current_dir),
                    capture_output=True,
                    text=True,
                )
            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if proc.returncode == 0:
                st.success("挖掘完成")
            else:
                st.error("挖掘失败")
            st.code(output if output else "(无输出)")

    out_dir = Path(current_dir) / "ml" / "saved_models" / "qlib_us"
    market_view = st.radio("查看市场", ["US", "CN"], horizontal=True, key="qlib_hub_view_market")
    out_dir = Path(current_dir) / "ml" / "saved_models" / f"qlib_{market_view.lower()}"

    summary_path = out_dir / "qlib_mining_summary_latest.json"
    factor_path = out_dir / "factor_mining_latest.csv"
    strategy_path = out_dir / "strategy_mining_latest.csv"
    segment_path = out_dir / "segment_strategy_compare_latest.csv"

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        if summary:
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("分层", str(summary.get("segment", "-")))
            x2.metric("因子数", int(summary.get("factor_rows", 0)))
            x3.metric("策略组合", int(summary.get("strategy_rows", 0)))
            top = (summary.get("top_strategies") or [{}])[0]
            x4.metric("最佳 Sharpe", f"{float(top.get('sharpe', 0.0)):.2f}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("因子排名")
        if factor_path.exists():
            st.dataframe(pd.read_csv(factor_path).head(30), use_container_width=True)
        else:
            st.caption("暂无因子结果")
    with col_r:
        st.subheader("策略排名")
        if strategy_path.exists():
            st.dataframe(pd.read_csv(strategy_path).head(30), use_container_width=True)
        else:
            st.caption("暂无策略结果")

    st.subheader("市值分层对比")
    if segment_path.exists():
        st.dataframe(pd.read_csv(segment_path), use_container_width=True)
    else:
        st.caption("暂无分层对比结果")


# --- V3 主导航 (精简版 4 入口) ---

st.sidebar.title("Coral Creek V3 🦅")
st.sidebar.caption("ML量化交易系统")

page = st.sidebar.radio("功能导航", [
    "🎯 每日机会",      # 原 每日工作台 + 买卖点 (行动中心)
    "📊 全量扫描",      # 原 每日扫描 (数据表)
    "🔬 个股研究",      # 原 个股分析 + 策略回测 (深度分析)
    "📰 新闻中心",      # 新闻 + 社交媒体 + AI 分类
    "💰 交易执行",      # 原 组合管理 + 策略实验室模拟盘 (Alpaca+Paper)
])

# 交易执行页默认按 US 上下文展示 Alpaca；其他页面会在各自 market 选择器里覆盖
if page == "💰 交易执行":
    _set_active_market("US")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Alpaca 持仓始终可见于左侧栏")
render_data_source_status_bar()
render_action_health_panel()

if page == "🎯 每日机会":
    # 整合: 每日工作台 + 买卖点信号
    render_todays_picks_page()
elif page == "📊 全量扫描":
    render_scan_page()
elif page == "🔬 个股研究":
    # 整合: 个股分析 + 策略回测 + AI中心(今日精选) + Qlib融合
    st.header("🔬 个股研究")
    research_tab = st.tabs(["🔍 个股分析", "🧪 策略回测", "🤖 AI选股", "🧠 Qlib融合"])
    with research_tab[0]:
        render_stock_lookup_page()
    with research_tab[1]:
        render_strategy_lab_page()
    with research_tab[2]:
        render_ai_center_page()
    with research_tab[3]:
        render_qlib_mining_hub()
elif page == "📰 新闻中心":
    from pages.news_center import render_news_center_page
    render_news_center_page()
elif page == "💰 交易执行":
    # 整合: 组合管理 (持仓+风控) + Paper Trading + Alpaca Trading
    render_portfolio_management_page()

