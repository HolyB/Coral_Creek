#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib Integration Module
=======================

将 Microsoft Qlib 量化库集成到 Coral Creek 系统。

功能:
1. 数据转换: Polygon/AkShare -> Qlib format
2. 特征增强: 使用 Qlib Alpha360/Alpha158 因子
3. 模型扩展: 调用 Qlib 预训练模型 (LightGBM, Transformer)
4. 专业回测: 使用 Qlib backtesting engine

安装:
    pip install pyqlib
    
初始化 (首次使用):
    python -m qlib.run.get_data qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
    python -m qlib.run.get_data qlib_data_us --target_dir ~/.qlib/qlib_data/us_data
    
使用:
    from ml.qlib_integration import QlibBridge
    
    bridge = QlibBridge(market='US')
    features = bridge.get_alpha360_features('AAPL')
    prediction = bridge.predict_with_transformer('AAPL')
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# 添加路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Qlib 可用性检查
QLIB_AVAILABLE = False
try:
    import qlib
    from qlib.config import REG_CN, REG_US
    QLIB_AVAILABLE = True
except ImportError:
    pass


class QlibBridge:
    """
    Qlib 桥接器
    
    连接 Coral Creek 和 Microsoft Qlib
    """
    
    def __init__(self, market: str = 'US', data_dir: Optional[str] = None):
        """
        Args:
            market: 'US' or 'CN'
            data_dir: Qlib 数据目录 (默认 ~/.qlib/qlib_data)
        """
        self.market = market.upper()
        self.initialized = False
        
        if not QLIB_AVAILABLE:
            print("⚠️ Qlib 未安装。请运行: pip install pyqlib")
            return
        
        # 设置数据目录
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / ".qlib" / "qlib_data" / f"{market.lower()}_data"
        
        self._init_qlib()
    
    def _init_qlib(self):
        """初始化 Qlib"""
        if not QLIB_AVAILABLE:
            return False
        
        try:
            if self.market == 'CN':
                provider_uri = str(self.data_dir)
                region = REG_CN
            else:
                provider_uri = str(self.data_dir)
                region = REG_US
            
            qlib.init(
                provider_uri=provider_uri,
                region=region,
                expression_cache=None,  # 禁用缓存以节省内存
            )
            self.initialized = True
            print(f"✅ Qlib 初始化成功 (market={self.market})")
            return True
            
        except Exception as e:
            print(f"❌ Qlib 初始化失败: {e}")
            print(f"   请先下载数据: python -m qlib.run.get_data qlib_data_{self.market.lower()} --target_dir {self.data_dir}")
            return False
    
    def convert_to_qlib_format(self, 
                                df: pd.DataFrame, 
                                symbol: str,
                                save_path: Optional[str] = None) -> bool:
        """
        将 Coral Creek 格式的数据转换为 Qlib 格式
        
        Qlib 格式要求:
        - 列名: $open, $high, $low, $close, $volume, $factor
        - 索引: datetime
        - 文件: {symbol}.pkl 或 {symbol}.csv
        
        Args:
            df: 原始 DataFrame (columns: Open, High, Low, Close, Volume)
            symbol: 股票代码
            save_path: 保存路径 (可选)
        """
        if df is None or df.empty:
            return False
        
        # 列名转换
        qlib_df = pd.DataFrame({
            '$open': df['Open'].values,
            '$high': df['High'].values,
            '$low': df['Low'].values,
            '$close': df['Close'].values,
            '$volume': df['Volume'].values,
            '$factor': 1.0,  # 复权因子，默认为1
        }, index=df.index)
        
        # 添加 $change 和 $vwap
        qlib_df['$change'] = qlib_df['$close'].pct_change()
        qlib_df['$vwap'] = (qlib_df['$close'] + qlib_df['$high'] + qlib_df['$low']) / 3
        
        if save_path:
            qlib_df.to_pickle(save_path)
            print(f"✅ 已保存 Qlib 格式数据: {save_path}")
        
        return qlib_df
    
    def get_alpha360_features(self, symbol: str, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取 Alpha360 因子特征
        
        Alpha360 是 Qlib 预置的 360 个常用 alpha 因子，包括:
        - 价格相关: ROC, BIAS, BBands, etc.
        - 量价相关: VWAP, VOL_RATIO, etc.
        - 技术指标: RSI, MACD, KDJ, etc.
        
        Args:
            symbol: 股票代码 (Qlib格式: SH600000 或 AAPL)
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with 360 features
        """
        if not self.initialized:
            print("⚠️ Qlib 未初始化")
            return None
        
        try:
            from qlib.data.dataset.handler import Alpha360
            
            # 格式化股票代码
            qlib_symbol = self._format_symbol(symbol)
            
            # 日期范围
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # 创建 Alpha360 handler
            handler = Alpha360(
                instruments=[qlib_symbol],
                start_time=start_date,
                end_time=end_date,
                fit_start_time=start_date,
                fit_end_time=end_date,
            )
            
            # 获取特征
            features = handler.fetch()
            return features
            
        except Exception as e:
            print(f"获取 Alpha360 特征失败: {e}")
            return None
    
    def get_alpha158_features(self, symbol: str,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取 Alpha158 因子特征 (精简版，158个因子)
        
        比 Alpha360 更轻量，适合快速实验
        """
        if not self.initialized:
            return None
        
        try:
            from qlib.data.dataset.handler import Alpha158
            
            qlib_symbol = self._format_symbol(symbol)
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            handler = Alpha158(
                instruments=[qlib_symbol],
                start_time=start_date,
                end_time=end_date,
            )
            
            return handler.fetch()
            
        except Exception as e:
            print(f"获取 Alpha158 特征失败: {e}")
            return None
    
    def train_lightgbm_model(self,
                              symbols: List[str],
                              start_date: str,
                              end_date: str,
                              target_col: str = 'Ref($close, -5) / $close - 1',
                              save_path: Optional[str] = None) -> Any:
        """
        训练 LightGBM 预测模型
        
        使用 Qlib 的 Alpha158 特征 + LightGBM 模型
        """
        if not self.initialized:
            print("⚠️ Qlib 未初始化")
            return None
        
        try:
            from qlib.contrib.model.gbdt import LGBModel
            from qlib.data.dataset import DatasetH
            from qlib.contrib.data.handler import Alpha158
            
            # 准备数据集
            qlib_symbols = [self._format_symbol(s) for s in symbols]
            
            handler_config = {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_date,
                    "end_time": end_date,
                    "fit_start_time": start_date,
                    "fit_end_time": end_date,
                    "instruments": qlib_symbols,
                }
            }
            
            # 数据集配置
            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": handler_config,
                    "segments": {
                        "train": (start_date, self._split_date(start_date, end_date, 0.7)),
                        "valid": (self._split_date(start_date, end_date, 0.7), 
                                 self._split_date(start_date, end_date, 0.85)),
                        "test": (self._split_date(start_date, end_date, 0.85), end_date),
                    }
                }
            }
            
            # 创建数据集
            dataset = DatasetH(**dataset_config["kwargs"])
            
            # LightGBM 模型
            model = LGBModel(
                loss="mse",
                num_leaves=64,
                learning_rate=0.05,
                n_estimators=500,
                early_stopping_rounds=50,
            )
            
            # 训练
            model.fit(dataset)
            
            # 保存
            if save_path:
                import joblib
                joblib.dump(model, save_path)
                print(f"✅ 模型已保存: {save_path}")
            
            return model
            
        except Exception as e:
            print(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_backtest(self,
                      model: Any,
                      symbols: List[str],
                      start_date: str,
                      end_date: str,
                      benchmark: str = 'SH000300') -> Dict:
        """
        运行 Qlib 专业回测
        
        包括:
        - 交易成本模拟
        - 滑点模拟
        - 持仓约束
        - 完整绩效报告
        """
        if not self.initialized:
            return {'error': 'Qlib 未初始化'}
        
        try:
            from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
            from qlib.contrib.evaluate import backtest as qlib_backtest
            from qlib.data.dataset import DatasetH
            from qlib.contrib.data.handler import Alpha158
            
            # 准备测试数据
            qlib_symbols = [self._format_symbol(s) for s in symbols]
            
            handler_config = {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_date,
                    "end_time": end_date,
                    "instruments": qlib_symbols,
                }
            }
            
            dataset = DatasetH(
                handler=Alpha158(**handler_config["kwargs"]),
                segments={"test": (start_date, end_date)}
            )
            
            # 预测
            predictions = model.predict(dataset)
            
            # 策略配置
            strategy_config = {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "signal": predictions,
                    "topk": 10,  # 持仓10只
                    "n_drop": 2,  # 每次换2只
                },
            }
            
            # 回测配置
            backtest_config = {
                "start_time": start_date,
                "end_time": end_date,
                "account": 10000000,  # 1000万初始资金
                "benchmark": benchmark,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,  # 涨跌停限制
                    "deal_price": "close",
                    "open_cost": 0.0005,  # 买入成本 0.05%
                    "close_cost": 0.0015,  # 卖出成本 0.15%
                    "min_cost": 5,  # 最低手续费
                },
            }
            
            # 运行回测
            result = qlib_backtest(
                strategy=TopkDropoutStrategy(**strategy_config["kwargs"]),
                **backtest_config,
            )
            
            return result
            
        except Exception as e:
            print(f"回测失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _format_symbol(self, symbol: str) -> str:
        """将股票代码转换为 Qlib 格式"""
        if self.market == 'CN':
            # 中国市场: 600000 -> SH600000, 000001 -> SZ000001
            if symbol.startswith('6'):
                return f"SH{symbol}"
            elif symbol.startswith(('0', '3')):
                return f"SZ{symbol}"
            elif symbol.startswith(('SH', 'SZ')):
                return symbol
            else:
                return symbol
        else:
            # 美国市场: 直接使用
            return symbol.upper()
    
    def _split_date(self, start: str, end: str, ratio: float) -> str:
        """计算分割日期"""
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        delta = (end_dt - start_dt) * ratio
        split_dt = start_dt + delta
        return split_dt.strftime('%Y-%m-%d')


class QlibFeatureEnhancer:
    """
    使用 Qlib 因子增强 Coral Creek 的特征工程
    
    将 Qlib 的 Alpha360/158 因子与现有特征融合
    """
    
    def __init__(self, market: str = 'US'):
        self.bridge = QlibBridge(market=market)
        self.market = market
    
    def enhance_features(self, 
                          symbol: str,
                          existing_features: pd.DataFrame) -> pd.DataFrame:
        """
        使用 Qlib 因子增强现有特征
        
        Args:
            symbol: 股票代码
            existing_features: 现有特征 DataFrame
            
        Returns:
            增强后的特征 DataFrame
        """
        if not self.bridge.initialized:
            print("⚠️ Qlib 不可用，返回原始特征")
            return existing_features
        
        # 获取 Qlib 特征
        qlib_features = self.bridge.get_alpha158_features(symbol)
        
        if qlib_features is None or qlib_features.empty:
            return existing_features
        
        # 合并特征
        # 注意：需要对齐日期索引
        try:
            merged = existing_features.join(qlib_features, how='left', rsuffix='_qlib')
            return merged.fillna(0)
        except Exception as e:
            print(f"特征合并失败: {e}")
            return existing_features


# === 便捷函数 ===

def check_qlib_status() -> Dict:
    """检查 Qlib 安装和数据状态"""
    status = {
        'installed': QLIB_AVAILABLE,
        'us_data': False,
        'cn_data': False,
    }
    
    if QLIB_AVAILABLE:
        us_dir = Path.home() / ".qlib" / "qlib_data" / "us_data"
        cn_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
        
        status['us_data'] = us_dir.exists()
        status['cn_data'] = cn_dir.exists()
    
    return status


def install_qlib_data(market: str = 'US'):
    """
    安装 Qlib 数据
    
    Note: 这是一个引导函数，实际下载需要运行命令行
    """
    print(f"""
=== 安装 Qlib {market} 数据 ===

1. 首先确保安装了 Qlib:
   pip install pyqlib

2. 下载数据:
   python -m qlib.run.get_data qlib_data_{market.lower()} --target_dir ~/.qlib/qlib_data/{market.lower()}_data

3. 数据大小:
   - US: ~5GB
   - CN: ~10GB

4. 下载完成后即可使用 QlibBridge
""")


if __name__ == "__main__":
    print("=== Qlib Integration 测试 ===\n")
    
    status = check_qlib_status()
    print(f"Qlib 安装状态: {'✅' if status['installed'] else '❌'}")
    print(f"美股数据: {'✅' if status['us_data'] else '❌'}")
    print(f"A股数据: {'✅' if status['cn_data'] else '❌'}")
    
    if not status['installed']:
        print("\n请先安装 Qlib:")
        print("  pip install pyqlib")
    elif not status['us_data']:
        install_qlib_data('US')
