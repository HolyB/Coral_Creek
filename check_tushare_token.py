#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare Token状态检查工具
检查API密钥是否过期、权限和积分状态
"""

import tushare as ts
import requests
import json
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从扫描脚本中读取的token
TUSHARE_TOKEN = 'gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482'

class TushareTokenChecker:
    """Tushare Token检查器"""
    
    def __init__(self, token=TUSHARE_TOKEN):
        self.token = token
        self.api_url = 'http://api.waditu.com/dataapi'
        
    def check_token_format(self):
        """检查token格式"""
        logger.info("🔍 检查Token格式...")
        
        if not self.token:
            logger.error("❌ Token为空")
            return False
            
        if len(self.token) < 30:
            logger.warning("⚠️ Token长度可能不正确")
            return False
            
        logger.info(f"✅ Token格式检查通过 (长度: {len(self.token)})")
        logger.info(f"Token前缀: {self.token[:10]}...")
        return True
    
    def test_token_basic_access(self):
        """测试Token基础访问权限"""
        logger.info("🔍 测试Token基础访问权限...")
        
        try:
            # 设置token
            ts.set_token(self.token)
            
            # 创建API实例，增加超时时间
            pro = ts.pro_api(token=self.token, timeout=60)
            
            # 尝试最简单的API调用 - 获取交易日历
            trade_cal = pro.trade_cal(
                exchange='SSE',
                start_date='20241201',
                end_date='20241210',
                fields='cal_date,is_open'
            )
            
            if not trade_cal.empty:
                logger.info("✅ Token基础访问权限正常")
                logger.info(f"测试数据: {len(trade_cal)}条交易日历记录")
                return True, pro
            else:
                logger.warning("⚠️ API返回空数据")
                return False, None
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Token访问测试失败: {error_msg}")
            
            # 分析具体错误
            if "认证失败" in error_msg or "token无效" in error_msg:
                logger.error("🔑 Token认证失败 - 可能已过期或无效")
            elif "积分不足" in error_msg:
                logger.error("💰 账户积分不足")
            elif "超出调用频率" in error_msg:
                logger.error("⏱️ API调用频率超限")
            elif "网络" in error_msg or "连接" in error_msg:
                logger.error("🌐 网络连接问题")
            
            return False, None
    
    def check_token_permissions(self, pro_api):
        """检查Token权限级别"""
        logger.info("🔍 检查Token权限级别...")
        
        permission_tests = [
            {
                'name': '股票基础信息',
                'test': lambda: pro_api.stock_basic(list_status='L', limit=5),
                'level': 'basic'
            },
            {
                'name': '日线行情数据',
                'test': lambda: pro_api.daily(ts_code='000001.SZ', start_date='20241201', end_date='20241205'),
                'level': 'basic'
            },
            {
                'name': '财务数据',
                'test': lambda: pro_api.income(ts_code='000001.SZ', period='20240930'),
                'level': 'advanced'
            },
            {
                'name': '分钟级数据',
                'test': lambda: pro_api.pro_bar(ts_code='000001.SZ', freq='1min', start_date='20241209', end_date='20241209'),
                'level': 'premium'
            }
        ]
        
        permissions = {'basic': False, 'advanced': False, 'premium': False}
        
        for test in permission_tests:
            try:
                logger.info(f"测试: {test['name']}")
                result = test['test']()
                
                if not result.empty:
                    logger.info(f"✅ {test['name']} - 权限正常")
                    permissions[test['level']] = True
                else:
                    logger.warning(f"⚠️ {test['name']} - 返回空数据")
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"❌ {test['name']} - 权限不足或其他错误: {error_msg}")
                
                if "权限不足" in error_msg or "需要" in error_msg:
                    logger.info(f"ℹ️ {test['name']} 需要更高级别权限")
        
        return permissions
    
    def get_user_info(self):
        """获取用户信息和积分状态"""
        logger.info("🔍 获取用户信息和积分状态...")
        
        try:
            # 直接调用API获取用户信息
            url = f"{self.api_url}/user"
            data = {
                'api_name': 'user',
                'token': self.token,
                'params': {},
                'fields': ''
            }
            
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('code') == 0:
                    user_data = result.get('data', {})
                    
                    if user_data.get('items'):
                        user_info = dict(zip(user_data['fields'], user_data['items'][0]))
                        
                        logger.info("✅ 用户信息获取成功:")
                        logger.info(f"   用户ID: {user_info.get('user_id', 'N/A')}")
                        logger.info(f"   当前积分: {user_info.get('point_total', 'N/A')}")
                        logger.info(f"   已用积分: {user_info.get('point_used', 'N/A')}")
                        logger.info(f"   剩余积分: {user_info.get('point_left', 'N/A')}")
                        logger.info(f"   到期时间: {user_info.get('exp_date', 'N/A')}")
                        
                        return user_info
                    else:
                        logger.warning("⚠️ 用户信息为空")
                else:
                    logger.error(f"❌ API返回错误: {result.get('msg', '未知错误')}")
            else:
                logger.error(f"❌ HTTP请求失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ 获取用户信息失败: {e}")
        
        return None
    
    def check_network_access(self):
        """检查网络访问状态"""
        logger.info("🔍 检查网络访问状态...")
        
        try:
            # 测试基础连通性
            response = requests.get("http://api.waditu.com", timeout=10)
            logger.info(f"✅ Tushare服务器可访问 (状态码: {response.status_code})")
            return True
        except Exception as e:
            logger.error(f"❌ 网络访问失败: {e}")
            return False
    
    def comprehensive_check(self):
        """综合检查"""
        print("=" * 60)
        print("🔧 Tushare Token状态综合检查")
        print("=" * 60)
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Token: {self.token[:10]}...{self.token[-6:]}")
        print("=" * 60)
        
        # 1. 检查Token格式
        if not self.check_token_format():
            return False
        
        # 2. 检查网络访问
        if not self.check_network_access():
            print("\n💡 建议:")
            print("- 检查网络连接")
            print("- 尝试使用VPN或代理")
            print("- 检查防火墙设置")
            return False
        
        # 3. 测试Token基础访问
        success, pro_api = self.test_token_basic_access()
        
        if not success:
            print("\n❌ Token验证失败")
            print("\n💡 可能的原因:")
            print("1. Token已过期")
            print("2. Token格式错误")
            print("3. 账户被暂停")
            print("4. 网络连接问题")
            print("\n🔧 解决方案:")
            print("1. 登录 https://tushare.pro 检查账户状态")
            print("2. 重新生成Token")
            print("3. 检查账户是否有欠费")
            return False
        
        # 4. 检查权限级别
        print("\n" + "="*30 + " 权限检查 " + "="*30)
        permissions = self.check_token_permissions(pro_api)
        
        print(f"\n📊 权限等级:")
        print(f"   基础权限: {'✅' if permissions['basic'] else '❌'}")
        print(f"   高级权限: {'✅' if permissions['advanced'] else '❌'}")
        print(f"   专业权限: {'✅' if permissions['premium'] else '❌'}")
        
        # 5. 获取用户信息
        print("\n" + "="*30 + " 用户信息 " + "="*30)
        user_info = self.get_user_info()
        
        if user_info:
            # 检查积分状态
            point_left = user_info.get('point_left', 0)
            if isinstance(point_left, (int, float)) and point_left <= 0:
                print("\n⚠️ 警告: 账户积分不足!")
                print("💡 请登录 https://tushare.pro 充值积分")
                
            # 检查到期时间
            exp_date = user_info.get('exp_date')
            if exp_date:
                try:
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    if exp_datetime < datetime.now():
                        print(f"\n⚠️ 警告: Token已过期 (到期时间: {exp_date})")
                        print("💡 请登录 https://tushare.pro 续费账户")
                except:
                    pass
        
        print("\n" + "="*60)
        print("🎉 Token状态检查完成!")
        
        if permissions['basic']:
            print("✅ Token可以正常使用基础功能")
            return True
        else:
            print("❌ Token无法正常使用")
            return False

def main():
    """主函数"""
    checker = TushareTokenChecker()
    success = checker.comprehensive_check()
    
    if not success:
        print("\n🔑 如何获取新的Token:")
        print("1. 访问 https://tushare.pro")
        print("2. 注册/登录账户")
        print("3. 在用户中心获取Token")
        print("4. 替换脚本中的TUSHARE_TOKEN变量")

if __name__ == "__main__":
    main() 