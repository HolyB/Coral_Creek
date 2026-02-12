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
            'symbol': result_dict.get('Symbol') or result_dict.get('symbol'),
            'scan_date': result_dict.get('Date') or result_dict.get('scan_date'),
            'price': result_dict.get('Price') or result_dict.get('price'),
            'turnover_m': result_dict.get('Turnover_M') or result_dict.get('turnover_m'),
            'blue_daily': result_dict.get('Blue_Daily') or result_dict.get('blue_daily'),
            'blue_weekly': result_dict.get('Blue_Weekly') or result_dict.get('blue_weekly'),
            'blue_monthly': result_dict.get('Blue_Monthly') or result_dict.get('blue_monthly'),
            'adx': result_dict.get('ADX') or result_dict.get('adx'),
            'volatility': result_dict.get('Volatility') or result_dict.get('volatility'),
            'is_heima': result_dict.get('Is_Heima') or result_dict.get('is_heima'),
            'is_juedi': result_dict.get('Is_Juedi') or result_dict.get('is_juedi'),
            'heima_daily': result_dict.get('Heima_Daily') or result_dict.get('heima_daily'),
            'heima_weekly': result_dict.get('Heima_Weekly') or result_dict.get('heima_weekly'),
            'heima_monthly': result_dict.get('Heima_Monthly') or result_dict.get('heima_monthly'),
            'juedi_daily': result_dict.get('Juedi_Daily') or result_dict.get('juedi_daily'),
            'juedi_weekly': result_dict.get('Juedi_Weekly') or result_dict.get('juedi_weekly'),
            'juedi_monthly': result_dict.get('Juedi_Monthly') or result_dict.get('juedi_monthly'),
            'market': result_dict.get('Market') or result_dict.get('market') or 'US',
            'company_name': result_dict.get('Company_Name') or result_dict.get('company_name'),
            'industry': result_dict.get('Industry') or result_dict.get('industry'),
            'market_cap': result_dict.get('Market_Cap') or result_dict.get('market_cap'),
            'cap_category': result_dict.get('Cap_Category') or result_dict.get('cap_category'),
            'day_high': result_dict.get('Day_High') or result_dict.get('day_high'),
            'day_low': result_dict.get('Day_Low') or result_dict.get('day_low'),
            'day_close': result_dict.get('Day_Close') or result_dict.get('day_close') or result_dict.get('Price') or result_dict.get('price'),
            # 补充缺失的字段
            'stop_loss': result_dict.get('Stop_Loss') or result_dict.get('stop_loss'),
            'strat_d_trend': result_dict.get('Strat_D_Trend') or result_dict.get('strat_d_trend'),
            'strat_c_resonance': result_dict.get('Strat_C_Resonance') or result_dict.get('strat_c_resonance'),
            'legacy_signal': result_dict.get('Legacy_Signal') or result_dict.get('legacy_signal'),
            'regime': result_dict.get('Regime') or result_dict.get('regime'),
            'adaptive_thresh': result_dict.get('Adaptive_Thresh') or result_dict.get('adaptive_thresh'),
            'vp_rating': result_dict.get('VP_Rating') or result_dict.get('vp_rating'),
            'profit_ratio': result_dict.get('Profit_Ratio') or result_dict.get('profit_ratio'),
            'wave_phase': result_dict.get('Wave_Phase') or result_dict.get('wave_phase'),
            'wave_desc': result_dict.get('Wave_Desc') or result_dict.get('wave_desc'),
            'chan_signal': result_dict.get('Chan_Signal') or result_dict.get('chan_signal'),
            'chan_desc': result_dict.get('Chan_Desc') or result_dict.get('chan_desc'),
            'duokongwang_buy': result_dict.get('Duokongwang_Buy') if result_dict.get('Duokongwang_Buy') is not None else result_dict.get('duokongwang_buy'),
            'duokongwang_sell': result_dict.get('Duokongwang_Sell') if result_dict.get('Duokongwang_Sell') is not None else result_dict.get('duokongwang_sell'),
        }
        
        # 移除 None 值的字段（Supabase 不接受某些 null）
        record = {k: v for k, v in record.items() if v is not None}
        
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
