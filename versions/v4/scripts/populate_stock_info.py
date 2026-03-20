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


def populate_us_stocks():
    """从 Polygon 获取所有美股基本信息并存入数据库"""
    import requests
    import time
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not found in environment variables")
        return
    
    print("📥 Fetching all US stocks from Polygon...")
    
    stock_list = []
    next_url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey={api_key}"
    
    page = 0
    while next_url:
        try:
            resp = requests.get(next_url, timeout=30)
            data = resp.json()
            
            if 'results' not in data:
                print(f"⚠️ No results in response: {data.get('error', 'unknown')}")
                break
            
            for item in data['results']:
                # 只包含美股 (NASDAQ, NYSE, etc.)
                if item.get('market') == 'stocks' and item.get('locale') == 'us':
                    stock_list.append({
                        'symbol': item.get('ticker', ''),
                        'name': item.get('name', ''),
                        'industry': item.get('sic_description', ''),
                        'area': '',
                        'market': 'US',
                        'list_date': ''
                    })
            
            page += 1
            print(f"   Page {page}: fetched {len(data['results'])} stocks (total: {len(stock_list)})")
            
            # 获取下一页
            next_url = data.get('next_url')
            if next_url and api_key not in next_url:
                next_url = f"{next_url}&apiKey={api_key}"
            
            # Rate limiting
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ Error fetching: {e}")
            break
    
    if not stock_list:
        print("❌ No US stocks fetched")
        return
    
    print(f"✅ Fetched {len(stock_list)} US stocks")
    
    print("💾 Saving to database...")
    bulk_upsert_stock_info(stock_list)
    
    count = get_stock_info_count(market='US')
    print(f"✅ Done! {count} US stocks in database")


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
    
    # 选择填充哪个市场
    import sys
    if len(sys.argv) > 1:
        market = sys.argv[1].upper()
        if market == 'CN':
            populate_cn_stocks()
        elif market == 'US':
            populate_us_stocks()
        elif market == 'ALL':
            populate_cn_stocks()
            print()
            populate_us_stocks()
        else:
            print(f"Unknown market: {market}. Use CN, US, or ALL")
    else:
        print("Usage: python populate_stock_info.py [CN|US|ALL]")
        print("  CN  - Populate A-share stocks from Tushare")
        print("  US  - Populate US stocks from Polygon")
        print("  ALL - Populate both markets")


if __name__ == "__main__":
    main()
