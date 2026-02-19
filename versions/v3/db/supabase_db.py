#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supabase 数据库连接层
用于替代 SQLite，支持云端数据访问
"""
import os
from datetime import datetime, date
from typing import List, Dict, Optional

# 尝试加载 Supabase
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

_supabase_client = None


def _first_non_none(*values):
    """返回第一个非 None 的值（保留 False/0/空字符串）。"""
    for v in values:
        if v is not None:
            return v
    return None


def _to_json_native(value):
    """
    将 numpy/pandas 标量转换为原生 Python 类型，避免
    `Object of type bool is not JSON serializable` 等错误。
    """
    if isinstance(value, dict):
        return {k: _to_json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_native(v) for v in value]

    # pandas/numpy 标量通常有 item()，转换为 Python 标量
    try:
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
    except Exception:
        pass

    return value


def get_supabase():
    """获取 Supabase 客户端"""
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    
    if not url or not key:
        return None
    
    if not SUPABASE_AVAILABLE:
        return None
    
    try:
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        print(f"⚠️ Supabase 连接失败: {e}")
        return None


def is_supabase_available() -> bool:
    """检查 Supabase 是否可用"""
    return get_supabase() is not None


def query_scan_results_supabase(scan_date: str = None, start_date: str = None, 
                                 end_date: str = None, min_blue: float = None, 
                                 market: str = None, limit: int = None) -> List[Dict]:
    """从 Supabase 查询扫描结果"""
    supabase = get_supabase()
    if not supabase:
        return []
    
    try:
        query = supabase.table('scan_results').select('*')
        
        if scan_date:
            query = query.eq('scan_date', scan_date)
        if start_date:
            query = query.gte('scan_date', start_date)
        if end_date:
            query = query.lte('scan_date', end_date)
        if min_blue is not None:
            query = query.gte('blue_daily', min_blue)
        if market:
            query = query.eq('market', market)
        
        query = query.order('scan_date', desc=True).order('blue_daily', desc=True)
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        return result.data or []
    except Exception as e:
        print(f"⚠️ Supabase 查询失败: {e}")
        return []


def get_scanned_dates_supabase(start_date: str = None, end_date: str = None, 
                                market: str = None) -> List[str]:
    """从 Supabase 获取已扫描日期"""
    supabase = get_supabase()
    if not supabase:
        return []
    
    try:
        query = supabase.table('scan_results').select('scan_date')
        
        if start_date:
            query = query.gte('scan_date', start_date)
        if end_date:
            query = query.lte('scan_date', end_date)
        if market:
            query = query.eq('market', market)
        
        result = query.execute()
        dates = list(set(r['scan_date'] for r in result.data))
        return sorted(dates, reverse=True)
    except Exception as e:
        print(f"⚠️ 获取日期失败: {e}")
        return []


def get_db_stats_supabase() -> Dict:
    """获取数据库统计"""
    supabase = get_supabase()
    if not supabase:
        return {'total_records': 0, 'total_symbols': 0, 'total_dates': 0}
    
    try:
        # 获取总记录数
        result = supabase.table('scan_results').select('id', count='exact').execute()
        total_records = result.count or 0
        
        # 获取日期范围
        dates_result = supabase.table('scan_results').select('scan_date').order('scan_date', desc=False).limit(1).execute()
        min_date = dates_result.data[0]['scan_date'] if dates_result.data else None
        
        dates_result = supabase.table('scan_results').select('scan_date').order('scan_date', desc=True).limit(1).execute()
        max_date = dates_result.data[0]['scan_date'] if dates_result.data else None
        
        return {
            'total_records': total_records,
            'min_date': min_date,
            'max_date': max_date,
            'source': 'supabase'
        }
    except Exception as e:
        print(f"⚠️ 统计失败: {e}")
        return {'total_records': 0, 'error': str(e)}


def insert_scan_result_supabase(result_dict: Dict) -> bool:
    """插入扫描结果到 Supabase - 完整字段版本"""
    supabase = get_supabase()
    if not supabase:
        return False
    
    try:
        record = {
            'symbol': _first_non_none(result_dict.get('Symbol'), result_dict.get('symbol')),
            'scan_date': _first_non_none(result_dict.get('Date'), result_dict.get('scan_date')),
            'price': _first_non_none(result_dict.get('Price'), result_dict.get('price')),
            'turnover_m': _first_non_none(result_dict.get('Turnover_M'), result_dict.get('turnover_m')),
            'blue_daily': _first_non_none(result_dict.get('Blue_Daily'), result_dict.get('blue_daily')),
            'blue_weekly': _first_non_none(result_dict.get('Blue_Weekly'), result_dict.get('blue_weekly')),
            'blue_monthly': _first_non_none(result_dict.get('Blue_Monthly'), result_dict.get('blue_monthly')),
            'adx': _first_non_none(result_dict.get('ADX'), result_dict.get('adx')),
            'volatility': _first_non_none(result_dict.get('Volatility'), result_dict.get('volatility')),
            'is_heima': _first_non_none(result_dict.get('Is_Heima'), result_dict.get('is_heima')),
            'is_juedi': _first_non_none(result_dict.get('Is_Juedi'), result_dict.get('is_juedi')),
            'heima_daily': _first_non_none(result_dict.get('Heima_Daily'), result_dict.get('heima_daily')),
            'heima_weekly': _first_non_none(result_dict.get('Heima_Weekly'), result_dict.get('heima_weekly')),
            'heima_monthly': _first_non_none(result_dict.get('Heima_Monthly'), result_dict.get('heima_monthly')),
            'juedi_daily': _first_non_none(result_dict.get('Juedi_Daily'), result_dict.get('juedi_daily')),
            'juedi_weekly': _first_non_none(result_dict.get('Juedi_Weekly'), result_dict.get('juedi_weekly')),
            'juedi_monthly': _first_non_none(result_dict.get('Juedi_Monthly'), result_dict.get('juedi_monthly')),
            'market': _first_non_none(result_dict.get('Market'), result_dict.get('market'), 'US'),
            'company_name': _first_non_none(result_dict.get('Company_Name'), result_dict.get('company_name')),
            'industry': _first_non_none(result_dict.get('Industry'), result_dict.get('industry')),
            'market_cap': _first_non_none(result_dict.get('Market_Cap'), result_dict.get('market_cap')),
            'cap_category': _first_non_none(result_dict.get('Cap_Category'), result_dict.get('cap_category')),
            'day_high': _first_non_none(result_dict.get('Day_High'), result_dict.get('day_high')),
            'day_low': _first_non_none(result_dict.get('Day_Low'), result_dict.get('day_low')),
            'day_close': _first_non_none(result_dict.get('Day_Close'), result_dict.get('day_close'), result_dict.get('Price'), result_dict.get('price')),
            # 补充缺失的字段
            'stop_loss': _first_non_none(result_dict.get('Stop_Loss'), result_dict.get('stop_loss')),
            'strat_d_trend': _first_non_none(result_dict.get('Strat_D_Trend'), result_dict.get('strat_d_trend')),
            'strat_c_resonance': _first_non_none(result_dict.get('Strat_C_Resonance'), result_dict.get('strat_c_resonance')),
            'legacy_signal': _first_non_none(result_dict.get('Legacy_Signal'), result_dict.get('legacy_signal')),
            'regime': _first_non_none(result_dict.get('Regime'), result_dict.get('regime')),
            'adaptive_thresh': _first_non_none(result_dict.get('Adaptive_Thresh'), result_dict.get('adaptive_thresh')),
            'vp_rating': _first_non_none(result_dict.get('VP_Rating'), result_dict.get('vp_rating')),
            'profit_ratio': _first_non_none(result_dict.get('Profit_Ratio'), result_dict.get('profit_ratio')),
            'wave_phase': _first_non_none(result_dict.get('Wave_Phase'), result_dict.get('wave_phase')),
            'wave_desc': _first_non_none(result_dict.get('Wave_Desc'), result_dict.get('wave_desc')),
            'chan_signal': _first_non_none(result_dict.get('Chan_Signal'), result_dict.get('chan_signal')),
            'chan_desc': _first_non_none(result_dict.get('Chan_Desc'), result_dict.get('chan_desc')),
            'duokongwang_buy': _first_non_none(result_dict.get('Duokongwang_Buy'), result_dict.get('duokongwang_buy')),
            'duokongwang_sell': _first_non_none(result_dict.get('Duokongwang_Sell'), result_dict.get('duokongwang_sell')),
        }
        
        # 移除 None 值的字段（Supabase 不接受某些 null）
        record = {k: v for k, v in record.items() if v is not None}
        # 转换 numpy/pandas 标量，确保 JSON 可序列化
        record = _to_json_native(record)
        
        try:
            supabase.table('scan_results').upsert(record, on_conflict='symbol,scan_date,market').execute()
            return True
        except Exception as e:
            msg = str(e).lower()
            if "day_high" in msg or "day_low" in msg or "day_close" in msg or "duokongwang_" in msg:
                # 兼容云端尚未迁移新列的场景：降级写入旧字段
                record.pop('day_high', None)
                record.pop('day_low', None)
                record.pop('day_close', None)
                record.pop('duokongwang_buy', None)
                record.pop('duokongwang_sell', None)
                supabase.table('scan_results').upsert(record, on_conflict='symbol,scan_date,market').execute()
                return True
            raise
    except Exception as e:
        print(f"⚠️ Supabase 插入失败: {e}")
        return False


def get_scan_date_counts_supabase(market: str = None, limit: int = 30) -> List[Dict]:
    """从 Supabase 获取每个扫描日期的记录数（轻量查询，仅取 scan_date 列）

    Returns:
        [{'scan_date': '2026-02-11', 'count': 275}, ...]  按日期降序
    """
    supabase = get_supabase()
    if not supabase:
        return []

    try:
        query = supabase.table('scan_results').select('scan_date')
        if market:
            query = query.eq('market', market)
        query = query.order('scan_date', desc=True)
        result = query.execute()
        if not result.data:
            return []

        # Python 端聚合计数
        from collections import Counter
        counter = Counter(r['scan_date'] for r in result.data)
        rows = [{'scan_date': d, 'count': c} for d, c in counter.items()]
        rows.sort(key=lambda x: x['scan_date'], reverse=True)
        return rows[:limit]
    except Exception as e:
        print(f"⚠️ Supabase 日期计数查询失败: {e}")
        return []


def get_first_scan_dates_supabase(symbols: List[str], market: str = 'US') -> Dict[str, str]:
    """从 Supabase 获取股票首次出现的日期
    
    Args:
        symbols: 股票代码列表
        market: 市场 (US/CN)
    
    Returns:
        dict: {symbol: first_scan_date}
    """
    supabase = get_supabase()
    if not supabase or not symbols:
        return {}
    
    try:
        # Supabase 不支持直接的 GROUP BY MIN，需要用 RPC 或分批查询
        # 这里用简单方法：查询所有匹配记录，在 Python 端处理
        result = supabase.table('scan_results')\
            .select('symbol, scan_date')\
            .eq('market', market)\
            .in_('symbol', symbols)\
            .order('scan_date', desc=False)\
            .execute()
        
        if not result.data:
            return {}
        
        # 取每个 symbol 的最早日期
        first_dates = {}
        for row in result.data:
            symbol = row['symbol']
            if symbol not in first_dates:
                first_dates[symbol] = row['scan_date']
        
        return first_dates
    except Exception as e:
        print(f"⚠️ 获取首次日期失败: {e}")
        return {}
