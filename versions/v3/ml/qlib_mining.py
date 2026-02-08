#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qlib 因子挖掘与策略挖掘工具。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.qlib_integration import QLIB_AVAILABLE, QlibBridge


DEFAULT_US_LARGE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "JPM", "XOM", "LLY", "V", "MA", "UNH", "COST",
]
DEFAULT_US_MID = [
    "SMCI", "PLTR", "HIMS", "RDDT", "SOFI", "HOOD", "APP", "ARM", "RIVN", "CAVA", "OKLO", "NNE", "IOT", "DKNG", "PINS",
]
DEFAULT_US_SMALL = [
    "IREN", "RIOT", "MARA", "CLSK", "BITF", "CORZ", "RUN", "FSLY", "UPST", "LMND", "SOUN", "BBAI", "IONQ", "SES", "JOBY",
]
DEFAULT_CN_LARGE = [
    "600519", "601318", "600036", "000333", "000858", "300750", "601899", "601398", "601288", "600887",
]


@dataclass
class MiningConfig:
    market: str
    symbols: List[str]
    start_date: str
    end_date: str
    topk_grid: List[int]
    drop_grid: List[int]
    segment: str = "CUSTOM"
    benchmark: Optional[str] = None
    min_cross_section: int = 5


class QlibMiningPipeline:
    """Qlib 挖掘管线：IC/IR 因子打分 + TopK/Dropout 策略网格。"""

    def __init__(self, config: MiningConfig, output_dir: Optional[Path] = None):
        self.config = config
        self.market = config.market.upper()
        self.output_dir = output_dir or (Path(__file__).parent / "saved_models" / f"qlib_{self.market.lower()}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = QlibBridge(market=self.market)

    def run(self) -> Dict:
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib 未安装，请先执行: pip install pyqlib")
        if not self.bridge.initialized:
            raise RuntimeError("Qlib 初始化失败，请先下载 market 对应数据")

        self._align_date_range_with_data()
        dataset = self._build_dataset()
        factor_df = self._mine_factors(dataset)
        strategy_df = self._mine_strategies(dataset)
        return self._save_reports(factor_df, strategy_df)

    def _align_date_range_with_data(self) -> None:
        cal_file = Path(self.bridge.data_dir) / "calendars" / "day.txt"
        if not cal_file.exists():
            return
        dates = [x.strip() for x in cal_file.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        if not dates:
            return
        min_d = dates[0]
        max_d = dates[-1]

        start = self.config.start_date
        end = self.config.end_date
        if end > max_d:
            end = max_d
        if start < min_d:
            start = min_d
        if start > end:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            start = max(min_d, (end_dt - timedelta(days=365)).strftime("%Y-%m-%d"))

        self.config.start_date = start
        self.config.end_date = end

    def _build_dataset(self):
        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH

        symbols = [self.bridge._format_symbol(s) for s in self.config.symbols]
        handler_cfg = {
            "start_time": self.config.start_date,
            "end_time": self.config.end_date,
            "fit_start_time": self.config.start_date,
            "fit_end_time": self.config.end_date,
            "instruments": symbols,
        }

        train_end = self.bridge._split_date(self.config.start_date, self.config.end_date, 0.7)
        valid_end = self.bridge._split_date(self.config.start_date, self.config.end_date, 0.85)

        return DatasetH(
            handler=Alpha158(**handler_cfg),
            segments={
                "train": (self.config.start_date, train_end),
                "valid": (train_end, valid_end),
                "test": (valid_end, self.config.end_date),
            },
        )

    @staticmethod
    def _to_df(obj) -> pd.DataFrame:
        if isinstance(obj, list):
            parts = [x for x in obj if isinstance(x, (pd.DataFrame, pd.Series))]
            if not parts:
                return pd.DataFrame()
            fixed = [p.to_frame() if isinstance(p, pd.Series) else p for p in parts]
            return pd.concat(fixed, axis=0)
        if isinstance(obj, pd.Series):
            return obj.to_frame()
        if isinstance(obj, pd.DataFrame):
            return obj
        return pd.DataFrame()

    @classmethod
    def _flatten_label(cls, label_obj: pd.DataFrame | pd.Series | list) -> pd.Series:
        label_df = cls._to_df(label_obj)
        if label_df.empty:
            return pd.Series(dtype=float)
        return label_df.iloc[:, 0]

    @classmethod
    def _flatten_feature(cls, feat_obj: pd.DataFrame | list) -> pd.DataFrame:
        feat_df = cls._to_df(feat_obj)
        if feat_df.empty:
            return pd.DataFrame()
        return feat_df

    @staticmethod
    def _flatten_prediction(pred_obj):
        if isinstance(pred_obj, list):
            parts = [x for x in pred_obj if isinstance(x, (pd.DataFrame, pd.Series))]
            if not parts:
                return pd.Series(dtype=float)
            fixed = []
            for p in parts:
                if isinstance(p, pd.DataFrame):
                    fixed.append(p.iloc[:, 0])
                else:
                    fixed.append(p)
            return pd.concat(fixed, axis=0)
        if isinstance(pred_obj, pd.DataFrame):
            return pred_obj.iloc[:, 0]
        return pred_obj

    @staticmethod
    def _flatten_label_legacy(label_obj: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(label_obj, pd.Series):
            return label_obj
        if not isinstance(label_obj, pd.DataFrame):
            raise TypeError(f"Unsupported label type: {type(label_obj)}")
        if label_obj.empty:
            return pd.Series(dtype=float)
        return label_obj.iloc[:, 0]

    @staticmethod
    def _get_datetime_level(idx: pd.MultiIndex) -> int:
        names = [n or "" for n in idx.names]
        for i, n in enumerate(names):
            if "date" in n.lower() or "time" in n.lower():
                return i
        return 0

    def _mine_factors(self, dataset) -> pd.DataFrame:
        from qlib.data.dataset.handler import DataHandlerLP

        feat_df = self._flatten_feature(
            dataset.prepare(["train", "valid"], col_set="feature", data_key=DataHandlerLP.DK_L)
        )
        if feat_df.empty:
            raise RuntimeError("Alpha158 特征为空，请检查时间区间是否超出本地 Qlib 数据范围")
        label_df = dataset.prepare(["train", "valid"], col_set="label", data_key=DataHandlerLP.DK_L)
        label_ser = self._flatten_label(label_df)

        if not isinstance(feat_df.index, pd.MultiIndex):
            raise RuntimeError("Alpha158 特征索引不是 MultiIndex，无法做截面 IC 分析")

        dt_level = self._get_datetime_level(feat_df.index)
        aligned_label = label_ser.reindex(feat_df.index)

        rows = []
        n_days_annual = 252.0

        for col in feat_df.columns:
            s = feat_df[col]
            daily_ic: List[float] = []
            grouped = s.groupby(level=dt_level)

            for dt, x in grouped:
                y = aligned_label.xs(dt, level=dt_level, drop_level=False)
                x_df = pd.concat([x, y.rename("label")], axis=1).dropna()
                if len(x_df) < self.config.min_cross_section:
                    continue
                ic = x_df.iloc[:, 0].corr(x_df["label"])
                if pd.notna(ic):
                    daily_ic.append(float(ic))

            if not daily_ic:
                continue

            ic_arr = np.asarray(daily_ic, dtype=float)
            ic_mean = float(np.nanmean(ic_arr))
            ic_std = float(np.nanstd(ic_arr))
            ir = float(ic_mean / ic_std * math.sqrt(n_days_annual)) if ic_std > 1e-12 else 0.0
            hit_rate = float(np.mean(ic_arr > 0))
            abs_ic = float(np.mean(np.abs(ic_arr)))

            rows.append(
                {
                    "factor": str(col),
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "ir": ir,
                    "hit_rate": hit_rate,
                    "abs_ic": abs_ic,
                    "sample_days": int(len(ic_arr)),
                }
            )

        result = pd.DataFrame(rows)
        if result.empty:
            return result

        result["score"] = (
            result["abs_ic"].rank(pct=True) * 0.5
            + result["ir"].abs().rank(pct=True) * 0.4
            + result["hit_rate"].rank(pct=True) * 0.1
        )
        result = result.sort_values(["score", "abs_ic"], ascending=False).reset_index(drop=True)
        return result

    def _mine_strategies(self, dataset) -> pd.DataFrame:
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data.dataset.handler import DataHandlerLP

        model = LGBModel(
            loss="mse",
            num_leaves=64,
            learning_rate=0.05,
            n_estimators=500,
            early_stopping_rounds=50,
        )
        model.fit(dataset)

        pred = self._flatten_prediction(model.predict(dataset, segment="test"))
        feat_test = self._flatten_feature(dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L))
        label_test = self._flatten_label(dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L))
        if feat_test.empty or label_test.empty:
            raise RuntimeError("测试集为空，无法进行策略挖掘")

        if isinstance(pred, (np.ndarray, list)):
            pred = pd.Series(np.asarray(pred).reshape(-1), index=feat_test.index[: len(np.asarray(pred).reshape(-1))])
        elif isinstance(pred, pd.Series):
            pred = pred.reindex(feat_test.index)
        else:
            raise RuntimeError(f"不支持的预测结果类型: {type(pred)}")

        panel = pd.DataFrame({"score": pred, "label": label_test.reindex(feat_test.index)}, index=feat_test.index).dropna()
        if panel.empty:
            raise RuntimeError("预测分数与标签对齐后为空")

        dt_level = self._get_datetime_level(panel.index)

        rows = []
        for topk in self.config.topk_grid:
            for n_drop in self.config.drop_grid:
                if n_drop >= topk:
                    continue
                metrics = self._simulate_topk_dropout(panel, dt_level=dt_level, topk=topk, n_drop=n_drop)
                rows.append(metrics)

        result = pd.DataFrame(rows)
        if result.empty:
            return result

        result["score"] = (
            result["ann_return"].rank(pct=True) * 0.45
            + result["sharpe"].rank(pct=True) * 0.35
            + (-result["max_drawdown"]).rank(pct=True) * 0.15
            + (-result["turnover"]).rank(pct=True) * 0.05
        )
        return result.sort_values("score", ascending=False).reset_index(drop=True)

    def _simulate_topk_dropout(self, panel: pd.DataFrame, dt_level: int, topk: int, n_drop: int) -> Dict:
        try:
            trading_days = list(panel.groupby(level=dt_level).groups.keys())
            holdings: List[str] = []
            daily_returns: List[float] = []
            turnover_list: List[float] = []

            for dt in trading_days:
                day_df = panel.xs(dt, level=dt_level)
                if day_df.empty:
                    continue
                day_df = day_df.sort_values("score", ascending=False)
                ranked = day_df.index.tolist()
                candidates = ranked[: max(topk * 2, topk)]

                if not holdings:
                    holdings = ranked[:topk]
                else:
                    keep_n = max(topk - n_drop, 0)
                    kept = [s for s in holdings if s in day_df.index][:keep_n]
                    replacements = [s for s in candidates if s not in kept][: max(topk - len(kept), 0)]
                    holdings = (kept + replacements)[:topk]

                if not holdings:
                    continue

                ret = float(day_df.loc[holdings, "label"].mean())
                daily_returns.append(ret)

                if len(daily_returns) == 1:
                    turnover_list.append(1.0)
                else:
                    prev_set = set(prev_holdings)
                    curr_set = set(holdings)
                    changed = len(curr_set - prev_set)
                    turnover_list.append(changed / max(len(curr_set), 1))

                prev_holdings = holdings.copy()

            if not daily_returns:
                raise RuntimeError("无有效交易日")

            daily = pd.Series(daily_returns, dtype=float)
            metrics = self._metrics_from_daily_returns(daily, turnover=float(np.mean(turnover_list)))
            metrics.update({"topk": int(topk), "n_drop": int(n_drop), "status": "ok", "error": ""})
            return metrics
        except Exception as exc:
            return {
                "topk": int(topk),
                "n_drop": int(n_drop),
                "ann_return": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "turnover": np.nan,
                "total_return": np.nan,
                "status": "error",
                "error": str(exc)[:180],
            }

    @staticmethod
    def _metrics_from_daily_returns(daily: pd.Series, turnover: float) -> Dict[str, float]:
        curve = (1.0 + daily).cumprod()
        n = max(len(daily), 1)
        ann_return = float(curve.iloc[-1] ** (252.0 / n) - 1.0)
        vol = float(daily.std())
        sharpe = float(daily.mean() / vol * math.sqrt(252.0)) if vol > 1e-12 else 0.0
        rolling_peak = curve.cummax()
        drawdown = (curve / rolling_peak) - 1.0
        max_dd = float(drawdown.min())
        return {
            "ann_return": ann_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": float(turnover),
            "total_return": float(curve.iloc[-1] - 1.0),
        }

    def _save_reports(self, factor_df: pd.DataFrame, strategy_df: pd.DataFrame) -> Dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        factor_csv = self.output_dir / f"factor_mining_{ts}.csv"
        strategy_csv = self.output_dir / f"strategy_mining_{ts}.csv"
        summary_json = self.output_dir / f"qlib_mining_summary_{ts}.json"

        factor_df.to_csv(factor_csv, index=False)
        strategy_df.to_csv(strategy_csv, index=False)

        top_factors = factor_df.head(20).to_dict(orient="records") if not factor_df.empty else []
        top_strategies = strategy_df.head(20).to_dict(orient="records") if not strategy_df.empty else []

        summary = {
            "created_at": datetime.now().isoformat(),
            "market": self.market,
            "segment": self.config.segment,
            "symbols": self.config.symbols,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "factor_rows": int(len(factor_df)),
            "strategy_rows": int(len(strategy_df)),
            "top_factors": top_factors[:10],
            "top_strategies": top_strategies[:10],
            "files": {
                "factor_csv": str(factor_csv),
                "strategy_csv": str(strategy_csv),
            },
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        latest_factor = self.output_dir / "factor_mining_latest.csv"
        latest_strategy = self.output_dir / "strategy_mining_latest.csv"
        latest_summary = self.output_dir / "qlib_mining_summary_latest.json"

        factor_df.to_csv(latest_factor, index=False)
        strategy_df.to_csv(latest_strategy, index=False)
        with open(latest_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


def run_segment_batch(
    market: str,
    segments: Sequence[str],
    start_date: str,
    end_date: str,
    topk_grid: Sequence[int],
    drop_grid: Sequence[int],
    benchmark: Optional[str] = None,
    min_cross_section: int = 5,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for seg in segments:
        symbols = build_symbol_pool(market, seg, symbols_arg=None)
        cfg = MiningConfig(
            market=market,
            segment=seg.upper(),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            topk_grid=list(topk_grid),
            drop_grid=list(drop_grid),
            benchmark=benchmark,
            min_cross_section=min_cross_section,
        )
        summary = QlibMiningPipeline(cfg, output_dir=output_dir).run()
        top_strategy = (summary.get("top_strategies") or [{}])[0]
        rows.append(
            {
                "segment": seg.upper(),
                "symbols_count": len(symbols),
                "best_topk": top_strategy.get("topk"),
                "best_n_drop": top_strategy.get("n_drop"),
                "best_ann_return": top_strategy.get("ann_return"),
                "best_sharpe": top_strategy.get("sharpe"),
                "best_max_drawdown": top_strategy.get("max_drawdown"),
                "best_turnover": top_strategy.get("turnover"),
                "best_total_return": top_strategy.get("total_return"),
                "status": top_strategy.get("status", "ok"),
            }
        )
    return pd.DataFrame(rows)


def build_symbol_pool(market: str, segment: str, symbols_arg: Optional[str] = None) -> List[str]:
    market = market.upper()
    segment = (segment or "ALL").upper()

    if symbols_arg:
        return [s.strip().upper() if market == "US" else s.strip() for s in symbols_arg.split(",") if s.strip()]

    if market == "CN":
        return DEFAULT_CN_LARGE.copy()

    mapping = {
        "LARGE": DEFAULT_US_LARGE,
        "MID": DEFAULT_US_MID,
        "SMALL": DEFAULT_US_SMALL,
        "ALL": DEFAULT_US_LARGE + DEFAULT_US_MID + DEFAULT_US_SMALL,
    }
    return mapping.get(segment, mapping["ALL"]).copy()


def parse_int_list(raw: str, default: Sequence[int]) -> List[int]:
    if not raw:
        return list(default)
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals or list(default)
