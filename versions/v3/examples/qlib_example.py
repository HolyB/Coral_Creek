#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib + Coral Creek 融合示例
===========================

展示如何使用 Qlib 增强 SmartPicker 的预测能力

使用场景:
1. 使用 Alpha158 替换/增强手工特征
2. 使用 LightGBM 排序模型提升选股精度
3. 使用 Qlib 回测引擎进行专业级回测

运行前准备:
    pip install pyqlib lightgbm
    python -m qlib.run.get_data qlib_data_us --target_dir ~/.qlib/qlib_data/us_data
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))


def example_1_check_environment():
    """示例 1: 检查 Qlib 环境"""
    print("=" * 60)
    print("示例 1: 检查 Qlib 环境")
    print("=" * 60)
    
    from ml.qlib_integration import check_qlib_status, install_qlib_data
    
    status = check_qlib_status()
    
    print(f"Qlib 已安装: {'✅' if status['installed'] else '❌'}")
    print(f"美股数据可用: {'✅' if status['us_data'] else '❌'}")
    print(f"A股数据可用: {'✅' if status['cn_data'] else '❌'}")
    
    if not status['installed']:
        print("\n👉 请先安装 Qlib: pip install pyqlib")
        return False
    
    if not status['us_data']:
        print("\n👉 请下载美股数据:")
        install_qlib_data('US')
        return False
    
    return True


def example_2_get_alpha_features():
    """示例 2: 获取 Alpha158 因子"""
    print("\n" + "=" * 60)
    print("示例 2: 获取 Alpha158 因子")
    print("=" * 60)
    
    from ml.qlib_integration import QlibBridge
    
    bridge = QlibBridge(market='US')
    
    if not bridge.initialized:
        print("⚠️ Qlib 未初始化，跳过")
        return
    
    # 获取 AAPL 的 Alpha158 特征
    features = bridge.get_alpha158_features('AAPL')
    
    if features is not None:
        print(f"特征数量: {len(features.columns)}")
        print(f"数据行数: {len(features)}")
        print(f"\n前10个特征:")
        for col in features.columns[:10]:
            print(f"  - {col}")
    else:
        print("获取特征失败")


def example_3_train_ranking_model():
    """示例 3: 训练 LightGBM 排序模型"""
    print("\n" + "=" * 60)
    print("示例 3: 训练 LightGBM 排序模型")
    print("=" * 60)
    
    from ml.qlib_integration import QlibBridge
    
    bridge = QlibBridge(market='US')
    
    if not bridge.initialized:
        print("⚠️ Qlib 未初始化，跳过")
        return None
    
    # 训练数据: SP500 部分股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'AVGO']
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2年数据
    
    print(f"训练数据: {len(symbols)} 只股票")
    print(f"时间范围: {start_date} ~ {end_date}")
    
    # 训练模型
    model = bridge.train_lightgbm_model(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        save_path=str(current_dir / "ml" / "saved_models" / "qlib_lgb_us.joblib")
    )
    
    if model:
        print("✅ 模型训练成功!")
    else:
        print("❌ 模型训练失败")
    
    return model


def example_4_compare_with_simple_backtest():
    """示例 4: 对比 Qlib 回测和简单回测"""
    print("\n" + "=" * 60)
    print("示例 4: 对比回测结果")
    print("=" * 60)
    
    # 使用现有的 SimpleBacktester
    from backtester import SimpleBacktester
    
    symbol = 'AAPL'
    
    # 1. 简单回测
    print("\n--- SimpleBacktester ---")
    bt = SimpleBacktester(symbol, market='US', days=365)
    if bt.load_data():
        bt.calculate_signals()
        bt.run_backtest()
        print(f"年化收益: {bt.results['Annual Return']:.2%}")
        print(f"最大回撤: {bt.results['Max Drawdown']:.2%}")
        print(f"胜率: {bt.results['Win Rate']:.2%}")
    
    # 2. Qlib 回测 (如果可用)
    try:
        from ml.qlib_integration import QlibBridge
        bridge = QlibBridge(market='US')
        
        if bridge.initialized:
            print("\n--- Qlib Backtest ---")
            # 这里需要先训练模型
            # result = bridge.run_backtest(model, [...], ...)
            print("(需要先运行 example_3 训练模型)")
    except Exception as e:
        print(f"Qlib 回测不可用: {e}")


def example_5_enhance_smart_picker():
    """示例 5: 使用 Qlib 增强 SmartPicker"""
    print("\n" + "=" * 60)
    print("示例 5: 增强 SmartPicker")
    print("=" * 60)
    
    from ml.smart_picker import SmartPicker
    from ml.qlib_integration import QlibFeatureEnhancer, QLIB_AVAILABLE
    
    # 创建 SmartPicker
    picker = SmartPicker(market='US')
    
    # 如果 Qlib 可用，创建增强器
    if QLIB_AVAILABLE:
        enhancer = QlibFeatureEnhancer(market='US')
        print("✅ Qlib 增强器已创建")
        print("   可用功能: Alpha158 特征, LightGBM 排序")
    else:
        print("⚠️ Qlib 不可用，使用基础 SmartPicker")
    
    # 测试选股
    import pandas as pd
    test_signals = pd.DataFrame([
        {'symbol': 'AAPL', 'price': 185.0, 'blue_daily': 125, 'blue_weekly': 110, 
         'blue_monthly': 90, 'is_heima': 1, 'company_name': 'Apple Inc'},
        {'symbol': 'MSFT', 'price': 420.0, 'blue_daily': 108, 'blue_weekly': 95, 
         'blue_monthly': 80, 'is_heima': 0, 'company_name': 'Microsoft'},
    ])
    
    picks = picker.pick(test_signals, {}, max_picks=2)
    
    print(f"\n推荐结果 ({len(picks)} 只):")
    for pick in picks:
        print(f"  {pick.symbol}: {pick.overall_score:.1f}分 | {'⭐' * pick.star_rating}")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Qlib + Coral Creek 融合示例                          ║
║                                                              ║
║  Qlib 是微软开源的 AI 量化投资平台，提供:                        ║
║  - 360+ Alpha 因子                                           ║
║  - 40+ 机器学习模型                                           ║
║  - 专业级回测引擎                                             ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # 1. 检查环境
    if not example_1_check_environment():
        print("\n⚠️ 环境检查未通过，后续示例可能无法运行")
        print("   但其他功能仍可正常使用。")
    
    # 2. 获取因子
    example_2_get_alpha_features()
    
    # 3. 训练模型 (可选，耗时较长)
    # model = example_3_train_ranking_model()
    
    # 4. 对比回测
    example_4_compare_with_simple_backtest()
    
    # 5. 增强选股
    example_5_enhance_smart_picker()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  下一步建议:                                                  ║
║                                                              ║
║  1. 安装 Qlib 数据:                                          ║
║     python -m qlib.run.get_data qlib_data_us                 ║
║                                                              ║
║  2. 训练排序模型:                                             ║
║     python examples/qlib_example.py --train                  ║
║                                                              ║
║  3. 运行对比回测:                                             ║
║     python examples/qlib_example.py --backtest               ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
