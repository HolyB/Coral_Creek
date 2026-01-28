#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coral Creek ML Module
机器学习模块 - 提供统计ML、深度学习和LLM功能
"""
from .feature_engineering import FeatureEngineer

# 模型相关 (延迟导入避免依赖问题)
def get_predictor(market: str = 'US'):
    """获取信号预测器"""
    from .predictor import get_predictor as _get_predictor
    return _get_predictor(market)

def get_registry():
    """获取模型注册中心"""
    from .model_registry import get_registry as _get_registry
    return _get_registry()
