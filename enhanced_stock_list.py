#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版A股股票列表获取工具
支持多个数据源，提供容错机制
"""

import pandas as pd
import numpy as np
import logging
import time
import requests
import json
from typing import List, Dict, Optional
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiSourceStockList:
    """多数据源A股股票列表获取器"""
    
    def __init__(self, cache_file='stock_list_cache.json', cache_hours=24):
        self.cache_file = cache_file
        self.cache_hours = cache_hours
        self.stock_list = []
        
    def get_stocks_from_tushare(self) -> pd.DataFrame:
        """从Tushare获取A股列表"""
        logger.info("🔍 尝试从Tushare获取A股列表...")
        try:
            import tushare as ts
            
            # 这里需要设置token
            TUSHARE_TOKEN = 'gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482'
            ts.set_token(TUSHARE_TOKEN)
            pro = ts.pro_api()
            
            # 获取A股股票基本信息
            stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
            
            if stock_info is None or stock_info.empty:
                logger.warning("❌ Tushare返回空数据")
                return pd.DataFrame()
            
            # 过滤A股（排除指数、债券等）
            stock_info = stock_info[stock_info['ts_code'].str.contains(r'\.(SH|SZ|BJ)$', regex=True)]
            
            # 转换格式
            tickers = []
            for _, row in stock_info.iterrows():
                tickers.append({
                    'code': row['ts_code'],  # tushare格式：600000.SH
                    'name': row['name'],
                    'industry': row.get('industry', ''),
                    'area': row.get('area', ''),
                    'source': 'tushare'
                })
            
            logger.info(f"✅ Tushare获取到 {len(tickers)} 只A股")
            return pd.DataFrame(tickers)
            
        except Exception as e:
            logger.error(f"❌ Tushare获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_akshare(self) -> pd.DataFrame:
        """从AKShare获取A股列表"""
        logger.info("🔍 尝试从AKShare获取A股列表...")
        try:
            import akshare as ak
            
            # 获取A股股票信息
            stock_info = ak.stock_info_a_code_name()
            
            if stock_info is None or stock_info.empty:
                logger.warning("❌ AKShare返回空数据")
                return pd.DataFrame()
            
            # 添加市场前缀
            tickers = []
            for _, row in stock_info.iterrows():
                code = row['code']
                name = row['name']
                
                # 转换为tushare格式
                if code.startswith('688') or code.startswith('6'):
                    ts_code = f'{code}.SH'
                elif code.startswith('3') or code.startswith('0'):
                    ts_code = f'{code}.SZ'
                elif code.startswith('8') or code.startswith('4'):
                    ts_code = f'{code}.BJ'
                else:
                    ts_code = f'{code}.SZ'
                
                tickers.append({
                    'code': ts_code,
                    'name': name,
                    'industry': '',
                    'area': '',
                    'source': 'akshare'
                })
            
            # 尝试获取北交所股票
            try:
                bj_stock_info = ak.stock_info_bj_name_code()
                for _, row in bj_stock_info.iterrows():
                    if '证券代码' in row and '证券简称' in row:
                        code = row['证券代码']
                        name = row['证券简称']
                        ts_code = f'{code}.BJ'
                        if not any(item['code'] == ts_code for item in tickers):
                            tickers.append({
                                'code': ts_code,
                                'name': name,
                                'industry': '',
                                'area': '',
                                'source': 'akshare_bj'
                            })
            except Exception as e:
                logger.warning(f"⚠️ 获取北交所股票失败: {e}")
            
            logger.info(f"✅ AKShare获取到 {len(tickers)} 只A股")
            return pd.DataFrame(tickers)
            
        except Exception as e:
            logger.error(f"❌ AKShare获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_eastmoney(self) -> pd.DataFrame:
        """从东方财富API获取A股列表"""
        logger.info("🔍 尝试从东方财富API获取A股列表...")
        try:
            # 东方财富股票列表API
            urls = [
                # 沪市A股
                'http://80.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:1+t:2,m:1+t:23&fields=f12,f14',
                # 深市A股
                'http://80.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80&fields=f12,f14',
                # 北交所
                'http://80.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:81&fields=f12,f14'
            ]
            
            all_tickers = []
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('data') and data['data'].get('diff'):
                            stocks = data['data']['diff']
                            for stock in stocks:
                                code = stock.get('f12', '')
                                name = stock.get('f14', '')
                                
                                if code and name:
                                    # 根据代码确定市场
                                    if code.startswith('6'):
                                        ts_code = f'{code}.SH'
                                    elif code.startswith('0') or code.startswith('3'):
                                        ts_code = f'{code}.SZ'
                                    elif code.startswith('8') or code.startswith('4'):
                                        ts_code = f'{code}.BJ'
                                    else:
                                        continue
                                    
                                    all_tickers.append({
                                        'code': ts_code,
                                        'name': name,
                                        'industry': '',
                                        'area': '',
                                        'source': 'eastmoney'
                                    })
                    
                    time.sleep(0.1)  # 避免请求过快
                    
                except Exception as e:
                    logger.warning(f"⚠️ 请求东方财富API失败: {e}")
                    continue
            
            logger.info(f"✅ 东方财富获取到 {len(all_tickers)} 只A股")
            return pd.DataFrame(all_tickers)
            
        except Exception as e:
            logger.error(f"❌ 东方财富API获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_sina(self) -> pd.DataFrame:
        """从新浪财经API获取A股列表"""
        logger.info("🔍 尝试从新浪财经API获取A股列表...")
        try:
            # 新浪财经股票列表API
            url = 'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=5000&sort=symbol&asc=1&node=hs_a'
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # 新浪返回的是JavaScript格式，需要处理
                text = response.text
                if text and text.startswith('[') and text.endswith(']'):
                    data = json.loads(text)
                    
                    tickers = []
                    for stock in data:
                        code = stock.get('code', '')
                        name = stock.get('name', '')
                        
                        if code and name:
                            # 新浪的格式需要转换
                            if code.startswith('sh'):
                                ts_code = f'{code[2:]}.SH'
                            elif code.startswith('sz'):
                                ts_code = f'{code[2:]}.SZ'
                            else:
                                continue
                            
                            tickers.append({
                                'code': ts_code,
                                'name': name,
                                'industry': '',
                                'area': '',
                                'source': 'sina'
                            })
                    
                    logger.info(f"✅ 新浪财经获取到 {len(tickers)} 只A股")
                    return pd.DataFrame(tickers)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ 新浪财经API获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_163(self) -> pd.DataFrame:
        """从网易财经API获取A股列表"""
        logger.info("🔍 尝试从网易财经API获取A股列表...")
        try:
            # 网易财经股票列表API
            urls = [
                'http://quotes.money.163.com/hs/service/diyrank.php?host=http%3A%2F%2Fquotes.money.163.com%2Fhs%2Fservice%2Fdiyrank.php&page=0&query=STYPE%3AEQA&fields=SYMBOL%2CNAME&sort=SYMBOL&order=asc&count=2000',
                'http://quotes.money.163.com/hs/service/diyrank.php?host=http%3A%2F%2Fquotes.money.163.com%2Fhs%2Fservice%2Fdiyrank.php&page=1&query=STYPE%3AEQA&fields=SYMBOL%2CNAME&sort=SYMBOL&order=asc&count=2000'
            ]
            
            all_tickers = []
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'list' in data:
                            for stock in data['list']:
                                code = stock.get('SYMBOL', '')
                                name = stock.get('NAME', '')
                                
                                if code and name and len(code) == 6:
                                    # 根据代码确定市场
                                    if code.startswith('6'):
                                        ts_code = f'{code}.SH'
                                    elif code.startswith(('0', '3')):
                                        ts_code = f'{code}.SZ'
                                    elif code.startswith(('8', '4')):
                                        ts_code = f'{code}.BJ'
                                    else:
                                        continue
                                    
                                    all_tickers.append({
                                        'code': ts_code,
                                        'name': name,
                                        'industry': '',
                                        'area': '',
                                        'source': 'netease'
                                    })
                    
                    time.sleep(0.1)  # 避免请求过快
                    
                except Exception as e:
                    logger.warning(f"⚠️ 网易财经API请求失败: {e}")
                    continue
            
            logger.info(f"✅ 网易财经获取到 {len(all_tickers)} 只A股")
            return pd.DataFrame(all_tickers)
            
        except Exception as e:
            logger.error(f"❌ 网易财经API获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_qq(self) -> pd.DataFrame:
        """从腾讯财经API获取A股列表"""
        logger.info("🔍 尝试从腾讯财经API获取A股列表...")
        try:
            # 腾讯财经股票列表API
            urls = [
                'http://qt.gtimg.cn/q=s_sh000001',  # 上证指数成分股
                'http://qt.gtimg.cn/q=s_sz399001',  # 深证成指成分股
            ]
            
            # 直接获取所有A股的方法
            base_url = 'http://qt.gtimg.cn/q='
            
            # 构建查询字符串（沪市）
            sh_codes = [f'sh{str(i).zfill(6)}' for i in range(600000, 605000, 10)]
            sz_codes = [f'sz{str(i).zfill(6)}' for i in range(1, 5000, 10)]
            
            all_tickers = []
            
            # 分批查询，避免URL过长
            batch_size = 50
            all_codes = sh_codes[:20] + sz_codes[:20]  # 先测试小批量
            
            for i in range(0, len(all_codes), batch_size):
                batch_codes = all_codes[i:i+batch_size]
                query = ','.join(batch_codes)
                url = f'{base_url}{query}'
                
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        content = response.text
                        lines = content.strip().split('\n')
                        
                        for line in lines:
                            if '~' in line and len(line.split('~')) > 1:
                                parts = line.split('~')
                                if len(parts) > 1:
                                    code = parts[0].split('=')[1]  # 提取代码
                                    name = parts[1] if len(parts) > 1 else ''
                                    
                                    if code and name and name != '':
                                        # 转换为tushare格式
                                        if code.startswith('sh'):
                                            ts_code = f'{code[2:]}.SH'
                                        elif code.startswith('sz'):
                                            ts_code = f'{code[2:]}.SZ'
                                        else:
                                            continue
                                        
                                        all_tickers.append({
                                            'code': ts_code,
                                            'name': name,
                                            'industry': '',
                                            'area': '',
                                            'source': 'tencent'
                                        })
                    
                    time.sleep(0.1)  # 避免请求过快
                    
                except Exception as e:
                    logger.warning(f"⚠️ 腾讯财经API请求失败: {e}")
                    continue
            
            logger.info(f"✅ 腾讯财经获取到 {len(all_tickers)} 只A股")
            return pd.DataFrame(all_tickers)
            
        except Exception as e:
            logger.error(f"❌ 腾讯财经API获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_cninfo(self) -> pd.DataFrame:
        """从巨潮资讯网获取A股列表"""
        logger.info("🔍 尝试从巨潮资讯网获取A股列表...")
        try:
            # 巨潮资讯网API
            url = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'http://www.cninfo.com.cn/',
            }
            
            # 获取股票代码列表的API
            stock_url = 'http://www.cninfo.com.cn/new/information/topSearch/query'
            
            response = requests.get(stock_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                all_tickers = []
                if 'stockList' in data:
                    for stock in data['stockList']:
                        code = stock.get('code', '')
                        name = stock.get('orgName', '') or stock.get('zwjc', '')
                        
                        if code and name and len(code) == 6:
                            # 根据代码确定市场
                            if code.startswith('6'):
                                ts_code = f'{code}.SH'
                            elif code.startswith(('0', '3')):
                                ts_code = f'{code}.SZ'
                            elif code.startswith(('8', '4')):
                                ts_code = f'{code}.BJ'
                            else:
                                continue
                            
                            all_tickers.append({
                                'code': ts_code,
                                'name': name,
                                'industry': '',
                                'area': '',
                                'source': 'cninfo'
                            })
                
                logger.info(f"✅ 巨潮资讯网获取到 {len(all_tickers)} 只A股")
                return pd.DataFrame(all_tickers)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ 巨潮资讯网API获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_csindex(self) -> pd.DataFrame:
        """从中证指数网站获取A股列表"""
        logger.info("🔍 尝试从中证指数网站获取A股列表...")
        try:
            # 中证指数公司API
            urls = [
                'https://www.csindex.com.cn/uploads/file/autofile/cons/000001cons.xls',  # 上证综指
                'https://www.csindex.com.cn/uploads/file/autofile/cons/399001cons.xls',  # 深证成指
            ]
            
            all_tickers = []
            
            for url in urls:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(url, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        # 这里应该解析Excel文件，但为了简单起见，我们跳过
                        # 实际使用时可以用pandas读取Excel
                        logger.info("✅ 中证指数数据源响应正常，但需要Excel解析")
                    
                except Exception as e:
                    logger.warning(f"⚠️ 中证指数请求失败: {e}")
                    continue
            
            # 由于需要解析Excel，这里返回空DataFrame
            # 实际应用中可以使用pandas读取Excel文件
            logger.info("✅ 中证指数获取到 0 只A股（需要Excel解析）")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ 中证指数API获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_static_list(self) -> pd.DataFrame:
        """从本地静态列表获取A股（备用方案）"""
        logger.info("🔍 尝试从本地静态列表获取A股...")
        try:
            # 一些常见的A股代码作为备用
            static_stocks = [
                # 沪市主要股票
                {'code': '000001.SZ', 'name': '平安银行'},
                {'code': '000002.SZ', 'name': '万科A'},
                {'code': '600000.SH', 'name': '浦发银行'},
                {'code': '600036.SH', 'name': '招商银行'},
                {'code': '600519.SH', 'name': '贵州茅台'},
                {'code': '600837.SH', 'name': '海通证券'},
                {'code': '600887.SH', 'name': '伊利股份'},
                {'code': '601318.SH', 'name': '中国平安'},
                {'code': '601398.SH', 'name': '工商银行'},
                {'code': '601857.SH', 'name': '中国石油'},
                # 深市主要股票
                {'code': '000858.SZ', 'name': '五粮液'},
                {'code': '002415.SZ', 'name': '海康威视'},
                {'code': '002594.SZ', 'name': '比亚迪'},
                {'code': '300014.SZ', 'name': '亿纬锂能'},
                {'code': '300015.SZ', 'name': '爱尔眼科'},
            ]
            
            for stock in static_stocks:
                stock['industry'] = ''
                stock['area'] = ''
                stock['source'] = 'static_backup'
            
            logger.info(f"✅ 本地静态列表获取到 {len(static_stocks)} 只A股")
            return pd.DataFrame(static_stocks)
            
        except Exception as e:
            logger.error(f"❌ 本地静态列表获取失败: {e}")
            return pd.DataFrame()
    
    def get_stocks_from_cache(self) -> pd.DataFrame:
        """从缓存文件获取股票列表"""
        try:
            if not os.path.exists(self.cache_file):
                return pd.DataFrame()
            
            # 检查缓存时间
            cache_time = os.path.getmtime(self.cache_file)
            current_time = time.time()
            if (current_time - cache_time) > (self.cache_hours * 3600):
                logger.info("📅 缓存已过期")
                return pd.DataFrame()
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            logger.info(f"✅ 从缓存获取到 {len(df)} 只A股")
            return df
            
        except Exception as e:
            logger.error(f"❌ 读取缓存失败: {e}")
            return pd.DataFrame()
    
    def save_to_cache(self, df: pd.DataFrame):
        """保存股票列表到缓存"""
        try:
            if not df.empty:
                data = df.to_dict('records')
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 缓存已保存: {len(df)} 只股票")
        except Exception as e:
            logger.error(f"❌ 保存缓存失败: {e}")
    
    def merge_stock_lists(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """合并多个数据源的股票列表"""
        if not dataframes or all(df.empty for df in dataframes):
            return pd.DataFrame()
        
        # 过滤空的DataFrame
        valid_dfs = [df for df in dataframes if not df.empty]
        
        if not valid_dfs:
            return pd.DataFrame()
        
        # 合并所有数据源
        all_stocks = pd.concat(valid_dfs, ignore_index=True)
        
        # 去重（以股票代码为准，保留第一个）
        merged_stocks = all_stocks.drop_duplicates(subset=['code'], keep='first')
        
        # 统计来源
        source_counts = all_stocks['source'].value_counts()
        logger.info("📊 数据源统计:")
        for source, count in source_counts.items():
            logger.info(f"   {source}: {count} 只股票")
        
        logger.info(f"🎯 合并后总计: {len(merged_stocks)} 只A股（去重后）")
        return merged_stocks.reset_index(drop=True)
    
    def get_stock_list(self, force_refresh=False) -> pd.DataFrame:
        """获取A股股票列表（主要入口函数）"""
        logger.info("=" * 60)
        logger.info("🚀 开始获取A股股票列表")
        logger.info("=" * 60)
        
        # 如果不强制刷新，先尝试缓存
        if not force_refresh:
            cached_df = self.get_stocks_from_cache()
            if not cached_df.empty:
                logger.info("🎉 使用缓存数据")
                return cached_df
        
        # 定义数据源获取函数
        data_sources = [
            ('Tushare', self.get_stocks_from_tushare),
            ('AKShare', self.get_stocks_from_akshare),
            ('东方财富', self.get_stocks_from_eastmoney),
            ('新浪财经', self.get_stocks_from_sina),
            ('网易财经', self.get_stocks_from_163),
            ('腾讯财经', self.get_stocks_from_qq),
            ('巨潮资讯', self.get_stocks_from_cninfo),
            ('中证指数', self.get_stocks_from_csindex),
            ('本地备用', self.get_stocks_from_static_list),
        ]
        
        # 尝试各个数据源
        results = []
        for source_name, get_func in data_sources:
            try:
                df = get_func()
                if not df.empty:
                    results.append(df)
                    logger.info(f"✅ {source_name} 数据获取成功")
                else:
                    logger.warning(f"⚠️ {source_name} 返回空数据")
            except Exception as e:
                logger.error(f"❌ {source_name} 获取失败: {e}")
        
        # 合并结果
        if results:
            final_df = self.merge_stock_lists(results)
            if not final_df.empty:
                # 保存到缓存
                self.save_to_cache(final_df)
                logger.info("🎉 A股列表获取完成")
                return final_df
        
        logger.error("❌ 所有数据源都失败了！")
        return pd.DataFrame()

def get_enhanced_cn_stock_list(force_refresh=False, cache_hours=24) -> pd.DataFrame:
    """
    获取增强版A股股票列表的便捷函数
    
    Args:
        force_refresh: 是否强制刷新（跳过缓存）
        cache_hours: 缓存有效期（小时）
    
    Returns:
        包含股票代码和名称的DataFrame
    """
    fetcher = MultiSourceStockList(cache_hours=cache_hours)
    return fetcher.get_stock_list(force_refresh=force_refresh)

def main():
    """测试函数"""
    print("测试多数据源A股列表获取...")
    
    # 测试强制刷新
    df = get_enhanced_cn_stock_list(force_refresh=True)
    
    if not df.empty:
        print(f"\n📊 获取结果:")
        print(f"总股票数: {len(df)}")
        print(f"数据源分布:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        print(f"\n前10只股票:")
        print(df[['code', 'name', 'source']].head(10))
        
        # 保存到CSV文件
        output_file = 'enhanced_stock_list.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果已保存到: {output_file}")
    else:
        print("❌ 未能获取到股票列表")

if __name__ == "__main__":
    main()
