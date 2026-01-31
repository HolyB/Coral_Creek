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
    """插入扫描结果到 Supabase"""
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
            'market': result_dict.get('Market') or result_dict.get('market') or 'US',
            'company_name': result_dict.get('Company_Name') or result_dict.get('company_name'),
            'industry': result_dict.get('Industry') or result_dict.get('industry'),
        }
        
        supabase.table('scan_results').upsert(record, on_conflict='symbol,scan_date,market').execute()
        return True
    except Exception as e:
        print(f"⚠️ 插入失败: {e}")
        return False
