#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qlib 自动挖掘入口脚本。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.qlib_mining import (  # noqa: E402
    MiningConfig,
    QlibMiningPipeline,
    build_symbol_pool,
    parse_int_list,
    run_segment_batch,
)


def main():
    parser = argparse.ArgumentParser(description="运行 Qlib 因子/策略挖掘")
    parser.add_argument("--market", default="US", choices=["US", "CN"], help="市场")
    parser.add_argument("--segment", default="ALL", choices=["ALL", "LARGE", "MID", "SMALL"], help="市值分层")
    parser.add_argument("--run-segment-batch", action="store_true", help="批量运行 LARGE/MID/SMALL 并输出对比")
    parser.add_argument("--symbols", default="", help="自定义股票池（逗号分隔，优先级高于 --segment）")
    parser.add_argument("--days", type=int, default=730, help="回溯天数")
    parser.add_argument("--topk-grid", default="5,8,10,15", help="策略 topk 网格")
    parser.add_argument("--drop-grid", default="1,2,3", help="策略 n_drop 网格")
    parser.add_argument("--benchmark", default="", help="基准代码（默认 US=SPY, CN=SH000300）")
    parser.add_argument("--min-cross-section", type=int, default=5, help="单日 IC 计算最小股票数")

    args = parser.parse_args()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    symbols = build_symbol_pool(args.market, args.segment, args.symbols or None)

    cfg = MiningConfig(
        market=args.market,
        segment=args.segment,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        topk_grid=parse_int_list(args.topk_grid, default=[5, 8, 10, 15]),
        drop_grid=parse_int_list(args.drop_grid, default=[1, 2, 3]),
        benchmark=args.benchmark or None,
        min_cross_section=args.min_cross_section,
    )

    print("\n" + "=" * 72)
    print("Qlib 自动挖掘")
    print("=" * 72)
    print(f"市场: {cfg.market}")
    print(f"市值分层: {args.segment}")
    print(f"股票数: {len(cfg.symbols)}")
    print(f"时间范围: {cfg.start_date} ~ {cfg.end_date}")
    print(f"topk 网格: {cfg.topk_grid}")
    print(f"n_drop 网格: {cfg.drop_grid}")

    try:
        if args.run_segment_batch and args.market.upper() == "US" and not args.symbols:
            batch_df = run_segment_batch(
                market=cfg.market,
                segments=["LARGE", "MID", "SMALL"],
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                topk_grid=cfg.topk_grid,
                drop_grid=cfg.drop_grid,
                benchmark=cfg.benchmark,
                min_cross_section=cfg.min_cross_section,
            )
            out_dir = project_root / "ml" / "saved_models" / f"qlib_{cfg.market.lower()}"
            out_dir.mkdir(parents=True, exist_ok=True)
            batch_path = out_dir / "segment_strategy_compare_latest.csv"
            batch_df.to_csv(batch_path, index=False)
            print("\n✅ 分层批量挖掘完成")
            print(f"分层对比: {batch_path}")
            print(batch_df.to_string(index=False))
            return

        pipeline = QlibMiningPipeline(config=cfg)
        summary = pipeline.run()
    except Exception as exc:
        print(f"\n❌ 挖掘失败: {exc}")
        print("\n建议检查：")
        print("1) 安装 Qlib: pip install pyqlib")
        print(
            "2) 下载数据: python -m qlib.run.get_data "
            f"qlib_data_{cfg.market.lower()} --target_dir ~/.qlib/qlib_data/{cfg.market.lower()}_data"
        )
        return

    print("\n✅ 挖掘完成")
    print(f"输出目录: {Path(summary['files']['factor_csv']).parent}")
    print(f"因子结果: {summary['files']['factor_csv']}")
    print(f"策略结果: {summary['files']['strategy_csv']}")

    if summary.get("top_factors"):
        best_factor = summary["top_factors"][0]
        print(f"最佳因子: {best_factor.get('factor')} | IC={best_factor.get('ic_mean'):.4f} | IR={best_factor.get('ir'):.4f}")

    if summary.get("top_strategies"):
        best_st = summary["top_strategies"][0]
        print(
            "最佳策略: "
            f"topk={best_st.get('topk')} n_drop={best_st.get('n_drop')} "
            f"ann={best_st.get('ann_return'):.2%} sharpe={best_st.get('sharpe'):.2f}"
        )


if __name__ == "__main__":
    main()
