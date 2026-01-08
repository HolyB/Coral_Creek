#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预填充股票基本信息到数据库
- A股: 从 Tushare 获取所有股票名称、行业
- 美股: 可以后续从 Polygon 获取
"""
import os
import sys
from dotenv import load_dotenv

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 加载 .env 文件 (从 versions/v2 目录)
load_dotenv(os.path.join(parent_dir, '.env'))

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import init_db, bulk_upsert_stock_info, get_stock_info_count


def populate_cn_stocks():
    """从 Tushare 获取所有 A 股基本信息并存入数据库"""
    import tushare as ts
    
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("❌ TUSHARE_TOKEN not found in environment variables")
        return
    
    ts.set_token(token)
    pro = ts.pro_api()
    
    print("📥 Fetching all CN A-share stock info from Tushare...")
    
    # 获取所有上市股票基本信息
    df = pro.stock_basic(
        exchange='', 
        list_status='L',
        fields='ts_code,symbol,name,area,industry,market,list_date'
    )
    
    if df is None or df.empty:
        print("❌ Failed to fetch stock info from Tushare")
        return
    
    print(f"✅ Fetched {len(df)} A-share stocks")
    
    # 转换为列表格式
    stock_list = []
    for _, row in df.iterrows():
        stock_list.append({
            'symbol': row['ts_code'],
            'name': row.get('name', ''),
            'industry': row.get('industry', ''),
            'area': row.get('area', ''),
            'market': 'CN',
            'list_date': row.get('list_date', '')
        })
    
    print("💾 Saving to database...")
    bulk_upsert_stock_info(stock_list)
    
    count = get_stock_info_count(market='CN')
    print(f"✅ Done! {count} CN stocks in database")


def main():
    print("🔧 Initializing database...")
    init_db()
    
    print("\n" + "="*50)
    print("  Stock Info Cache Population Script")
    print("="*50 + "\n")
    
    # 当前状态
    cn_count = get_stock_info_count(market='CN')
    us_count = get_stock_info_count(market='US')
    print(f"📊 Current stock_info status:")
    print(f"   - CN (A-shares): {cn_count}")
    print(f"   - US (US stocks): {us_count}")
    print()
    
    # 填充 A 股
    populate_cn_stocks()


if __name__ == "__main__":
    main()
