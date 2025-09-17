#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare连接诊断和修复工具
用于解决API连接超时和网络问题
"""

import tushare as ts
import requests
import time
import socket
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import logging
import pandas as pd # Added missing import for pandas

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tushare配置
TUSHARE_TOKEN = 'gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482'

class TushareConnectionFix:
    """Tushare连接修复器"""
    
    def __init__(self, token=TUSHARE_TOKEN):
        self.token = token
        self.base_urls = [
            'http://api.waditu.com/dataapi',
            'https://api.waditu.com/dataapi',  # 尝试HTTPS
        ]
        self.backup_servers = [
            '103.26.0.2',
            '103.26.0.3', 
            '103.26.0.4',
            '103.26.0.5'
        ]
    
    def test_network_connectivity(self):
        """测试网络连通性"""
        logger.info("🔍 开始网络连通性测试...")
        
        # 测试DNS解析
        try:
            socket.gethostbyname('api.waditu.com')
            logger.info("✅ DNS解析正常")
        except Exception as e:
            logger.error(f"❌ DNS解析失败: {e}")
            return False
        
        # 测试端口连通性
        for ip in self.backup_servers:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((ip, 80))
                sock.close()
                if result == 0:
                    logger.info(f"✅ 端口80连通 - IP: {ip}")
                else:
                    logger.warning(f"⚠️ 端口80不通 - IP: {ip}")
            except Exception as e:
                logger.error(f"❌ 连接测试失败 - IP: {ip}, 错误: {e}")
        
        return True
    
    def create_robust_session(self, timeout=60):
        """创建健壮的HTTP会话"""
        session = requests.Session()
        
        # 设置重试策略
        retry_strategy = Retry(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=2,
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置更长的超时时间
        session.timeout = timeout
        
        # 设置User-Agent
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        return session
    
    def test_api_endpoints(self):
        """测试API端点"""
        logger.info("🔍 测试API端点...")
        session = self.create_robust_session()
        
        for url in self.base_urls:
            try:
                logger.info(f"测试端点: {url}")
                
                # 构造测试请求
                test_params = {
                    'api_name': 'stock_basic',
                    'token': self.token,
                    'params': {'list_status': 'L', 'limit': 10},
                    'fields': 'ts_code,symbol,name'
                }
                
                response = session.post(f"{url}/stock_basic", 
                                      json=test_params, 
                                      timeout=60)
                
                if response.status_code == 200:
                    logger.info(f"✅ API端点可用: {url}")
                    return url
                else:
                    logger.warning(f"⚠️ API端点返回错误状态: {url} - {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ API端点测试失败: {url} - {e}")
        
        return None
    
    def patch_tushare_timeout(self, timeout=120):
        """修补Tushare的超时设置"""
        logger.info(f"🔧 修补Tushare超时设置为 {timeout} 秒...")
        
        try:
            # 设置token
            ts.set_token(self.token)
            
            # 创建带更长超时的API实例
            pro = ts.pro_api(token=self.token, timeout=timeout)
            
            logger.info("✅ Tushare API实例创建成功")
            return pro
            
        except Exception as e:
            logger.error(f"❌ Tushare API创建失败: {e}")
            return None
    
    def test_stock_basic_api(self, pro_api):
        """测试获取股票基础信息"""
        logger.info("🔍 测试股票基础信息API...")
        
        try:
            # 分批获取，避免超时
            stock_list = pro_api.stock_basic(
                exchange='',
                list_status='L',
                limit=50,  # 限制数量
                fields='ts_code,symbol,name,area,industry,list_date'
            )
            
            if not stock_list.empty:
                logger.info(f"✅ 成功获取 {len(stock_list)} 只股票信息")
                logger.info(f"示例: {stock_list.head(3).to_string()}")
                return True
            else:
                logger.warning("⚠️ 获取到空的股票列表")
                return False
                
        except Exception as e:
            logger.error(f"❌ 股票基础信息API测试失败: {e}")
            return False
    
    def apply_connection_fix(self):
        """应用连接修复"""
        logger.info("🚀 开始Tushare连接修复...")
        
        # 1. 测试网络
        if not self.test_network_connectivity():
            logger.error("❌ 网络连通性测试失败，请检查网络设置")
            return None
        
        # 2. 测试API端点
        working_endpoint = self.test_api_endpoints()
        if not working_endpoint:
            logger.error("❌ 所有API端点都无法访问")
            return None
        
        # 3. 创建修复后的API实例
        pro_api = self.patch_tushare_timeout(timeout=120)
        if not pro_api:
            return None
        
        # 4. 测试API功能
        if self.test_stock_basic_api(pro_api):
            logger.info("🎉 Tushare连接修复成功！")
            return pro_api
        else:
            logger.error("❌ API功能测试失败")
            return None
    
    def get_stocks_with_retry(self, pro_api, batch_size=100, max_retries=3):
        """带重试机制的股票列表获取"""
        logger.info("📈 开始获取完整股票列表...")
        
        all_stocks = []
        offset = 0
        
        while True:
            for retry in range(max_retries):
                try:
                    logger.info(f"获取批次 {offset//batch_size + 1}，起始位置: {offset}")
                    
                    batch_stocks = pro_api.stock_basic(
                        exchange='',
                        list_status='L',
                        offset=offset,
                        limit=batch_size,
                        fields='ts_code,symbol,name,area,industry,list_date,market'
                    )
                    
                    if batch_stocks.empty:
                        logger.info("✅ 所有股票数据获取完成")
                        return pd.concat(all_stocks, ignore_index=True) if all_stocks else pd.DataFrame()
                    
                    all_stocks.append(batch_stocks)
                    offset += batch_size
                    
                    logger.info(f"✅ 成功获取 {len(batch_stocks)} 只股票，累计: {sum(len(df) for df in all_stocks)}")
                    
                    # 避免API频率限制
                    time.sleep(0.5)
                    break
                    
                except Exception as e:
                    logger.warning(f"⚠️ 批次获取失败 (重试 {retry+1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)  # 指数退避
                    else:
                        logger.error(f"❌ 批次获取最终失败，跳过位置 {offset}")
                        offset += batch_size
                        break
        
        return pd.concat(all_stocks, ignore_index=True) if all_stocks else pd.DataFrame()

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 Tushare连接诊断和修复工具")
    print("=" * 60)
    
    # 创建修复器
    fixer = TushareConnectionFix()
    
    # 应用修复
    pro_api = fixer.apply_connection_fix()
    
    if pro_api:
        print("\n" + "=" * 60)
        print("📊 获取完整股票列表示例")
        print("=" * 60)
        
        # 获取股票列表
        stocks = fixer.get_stocks_with_retry(pro_api, batch_size=200)
        
        if not stocks.empty:
            print(f"✅ 总共获取 {len(stocks)} 只股票")
            print("\n前10只股票示例:")
            print(stocks.head(10).to_string(index=False))
            
            # 保存到文件
            filename = f"stock_list_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            stocks.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n💾 股票列表已保存到: {filename}")
        else:
            print("❌ 未能获取股票列表")
    else:
        print("\n❌ Tushare连接修复失败")
        print("\n🔧 建议的解决方案:")
        print("1. 检查网络连接和防火墙设置")
        print("2. 尝试使用VPN或代理")
        print("3. 联系网络管理员检查企业网络限制")
        print("4. 稍后重试，可能是服务器临时问题")
        print("5. 检查Tushare token是否有效")

if __name__ == "__main__":
    main() 