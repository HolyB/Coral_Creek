#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib 模型训练与导出脚本
======================

本地训练 Qlib 模型，导出为可部署的格式。

使用流程:
1. 本地安装 Qlib 和数据
2. 运行此脚本训练模型
3. 模型会保存到 ml/saved_models/qlib_*/
4. 提交模型文件到 Git (或上传到云存储)
5. 线上使用 inference_only=True 模式

用法:
    python scripts/train_qlib_model.py --market US --symbols SP500
    python scripts/train_qlib_model.py --market CN --symbols CSI300
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 预定义股票池
STOCK_POOLS = {
    'SP500_TOP50': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
        'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'DHR',
        'ABT', 'VZ', 'ADBE', 'CRM', 'NKE', 'CMCSA', 'NEE', 'TXN', 'PM', 'INTC',
        'RTX', 'ORCL', 'AMD', 'HON', 'QCOM', 'BA', 'UPS', 'IBM', 'LOW', 'CAT'
    ],
    'TECH_TOP20': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'AVGO',
        'ADBE', 'CRM', 'ORCL', 'CSCO', 'QCOM', 'TXN', 'IBM', 'AMAT', 'NOW', 'INTU'
    ],
    'CSI300_SAMPLE': [
        '600000', '600036', '600519', '600887', '601318', '000001', '000002', '000333',
        '000651', '000858', '002415', '002594', '300750', '601166', '601288'
    ],
}


def train_and_export(market: str, 
                     symbols: list,
                     start_date: str,
                     end_date: str,
                     output_dir: Path) -> bool:
    """
    训练并导出模型
    """
    from ml.qlib_integration import QlibBridge, QLIB_AVAILABLE
    
    if not QLIB_AVAILABLE:
        print("❌ 请先安装 Qlib: pip install pyqlib")
        return False
    
    print(f"\n{'='*60}")
    print(f"训练 Qlib 模型")
    print(f"{'='*60}")
    print(f"市场: {market}")
    print(f"股票数: {len(symbols)}")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    # 初始化 Qlib Bridge
    bridge = QlibBridge(market=market)
    
    if not bridge.initialized:
        print("❌ Qlib 初始化失败")
        return False
    
    # 训练模型
    print("开始训练 LightGBM 模型...")
    
    model_path = output_dir / "lightgbm_ranker.joblib"
    model = bridge.train_lightgbm_model(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        save_path=str(model_path)
    )
    
    if model is None:
        print("❌ 模型训练失败")
        return False
    
    # 保存元数据
    metadata = {
        'market': market,
        'symbols': symbols,
        'train_start': start_date,
        'train_end': end_date,
        'created_at': datetime.now().isoformat(),
        'model_type': 'LightGBM',
        'feature_set': 'Alpha158',
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 保存特征名 (用于推理时对齐特征)
    # 注意: 这里需要从 Alpha158 handler 获取特征名
    try:
        from qlib.contrib.data.handler import Alpha158
        handler = Alpha158(
            instruments=symbols[:1],  # 只用一个股票获取特征名
            start_time=end_date,
            end_time=end_date,
        )
        feature_names = list(handler.fetch().columns)
        
        with open(output_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        print(f"✅ 特征配置已保存 ({len(feature_names)} 个特征)")
    except Exception as e:
        print(f"⚠️ 保存特征名失败: {e}")
    
    print(f"\n✅ 模型已导出到: {output_dir}")
    print(f"   - lightgbm_ranker.joblib")
    print(f"   - metadata.json")
    print(f"   - feature_names.json")
    
    return True


def upload_to_cloud(output_dir: Path, bucket_name: str = None):
    """
    上传模型到云存储 (可选)
    
    支持:
    - S3
    - GCS
    - Supabase Storage
    """
    print("\n上传到云存储...")
    
    # 尝试 Supabase
    try:
        from db.supabase_db import get_supabase_client
        supabase = get_supabase_client()
        
        if supabase:
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        supabase.storage.from_('models').upload(
                            f"qlib/{file_path.name}",
                            f.read()
                        )
                    print(f"  ✓ 上传: {file_path.name}")
            print("✅ 模型已上传到 Supabase Storage")
            return True
    except Exception as e:
        print(f"⚠️ Supabase 上传失败: {e}")
    
    # 提示手动上传
    print("""
    💡 也可以手动上传到:
    - GitHub Release (推荐小于 50MB)
    - S3 / GCS / Azure Blob
    - Hugging Face Hub
    """)
    
    return False


def main():
    parser = argparse.ArgumentParser(description='训练并导出 Qlib 模型')
    parser.add_argument('--market', default='US', choices=['US', 'CN'], help='市场')
    parser.add_argument('--symbols', default='TECH_TOP20', 
                        help='股票池 (SP500_TOP50/TECH_TOP20/CSI300_SAMPLE 或逗号分隔的代码)')
    parser.add_argument('--days', type=int, default=730, help='训练数据天数')
    parser.add_argument('--upload', action='store_true', help='训练后上传到云存储')
    
    args = parser.parse_args()
    
    # 解析股票池
    if args.symbols in STOCK_POOLS:
        symbols = STOCK_POOLS[args.symbols]
    else:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # 时间范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # 输出目录
    output_dir = project_root / "ml" / "saved_models" / f"qlib_{args.market.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    success = train_and_export(
        market=args.market,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    if success and args.upload:
        upload_to_cloud(output_dir)
    
    print("\n下一步:")
    print(f"  1. 检查模型: ls -la {output_dir}")
    print(f"  2. 提交到 Git: git add {output_dir} && git commit")
    print(f"  3. 线上使用: QlibBridge(market='{args.market}', inference_only=True)")


if __name__ == "__main__":
    main()
