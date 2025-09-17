#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络连接诊断工具
用于测试Tushare API的网络连接状况
"""

import requests
import socket
import time
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError

def test_dns_resolution():
    """测试DNS解析"""
    print("🔍 测试DNS解析...")
    try:
        ip = socket.gethostbyname('api.waditu.com')
        print(f"✅ DNS解析成功: api.waditu.com -> {ip}")
        return True, ip
    except socket.gaierror as e:
        print(f"❌ DNS解析失败: {e}")
        return False, None

def test_tcp_connection(host, port=80):
    """测试TCP连接"""
    print(f"🔍 测试TCP连接到 {host}:{port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ TCP连接成功到 {host}:{port}")
            return True
        else:
            print(f"❌ TCP连接失败到 {host}:{port} (错误代码: {result})")
            return False
    except Exception as e:
        print(f"❌ TCP连接异常: {e}")
        return False

def test_http_requests():
    """测试HTTP请求"""
    print("🔍 测试HTTP请求...")
    
    test_urls = [
        "http://api.waditu.com",
        "https://tushare.pro",
        "http://www.baidu.com"  # 对照测试
    ]
    
    for url in test_urls:
        try:
            print(f"  测试: {url}")
            response = requests.get(url, timeout=10)
            print(f"    ✅ 状态码: {response.status_code}")
        except requests.exceptions.ConnectTimeout:
            print(f"    ❌ 连接超时")
        except requests.exceptions.ConnectionError as e:
            print(f"    ❌ 连接错误: {e}")
        except Exception as e:
            print(f"    ❌ 其他错误: {e}")

def test_tushare_api():
    """测试Tushare API调用"""
    print("🔍 测试Tushare API调用...")
    
    try:
        # 模拟API调用
        api_url = 'http://api.waditu.com/dataapi'
        data = {
            'api_name': 'trade_cal',
            'token': 'test_token',
            'params': {'exchange': 'SSE', 'start_date': '20241201', 'end_date': '20241202'},
            'fields': 'cal_date,is_open'
        }
        
        print(f"  请求URL: {api_url}")
        response = requests.post(api_url, json=data, timeout=15)
        print(f"  ✅ API响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  📄 API响应内容: {result}")
        
    except Exception as e:
        print(f"  ❌ API调用失败: {e}")

def check_proxy_settings():
    """检查代理设置"""
    print("🔍 检查代理设置...")
    
    import os
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  🔧 {var}: {value}")
        else:
            print(f"  ➖ {var}: 未设置")

def main():
    """主诊断函数"""
    print("=" * 60)
    print("🩺 Tushare网络连接诊断工具")
    print("=" * 60)
    print(f"诊断时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. DNS解析测试
    dns_ok, ip = test_dns_resolution()
    print()
    
    # 2. TCP连接测试
    if dns_ok:
        tcp_ok = test_tcp_connection(ip)
    else:
        tcp_ok = test_tcp_connection('103.26.0.5')  # 直接使用IP
    print()
    
    # 3. HTTP请求测试
    test_http_requests()
    print()
    
    # 4. 代理设置检查
    check_proxy_settings()
    print()
    
    # 5. Tushare API测试
    test_tushare_api()
    print()
    
    print("=" * 60)
    print("🎯 诊断建议:")
    
    if not dns_ok:
        print("❌ DNS解析问题 - 检查DNS设置或网络连接")
    elif not tcp_ok:
        print("❌ TCP连接问题 - 可能是防火墙阻止或网络限制")
        print("💡 建议:")
        print("  1. 检查防火墙设置")
        print("  2. 尝试使用VPN")
        print("  3. 联系网络管理员")
    else:
        print("✅ 基础网络连接正常")
        print("💡 如果Tushare仍无法使用，可能是:")
        print("  1. Tushare服务器临时不可用")
        print("  2. 需要特定的网络配置")
        print("  3. Token确实有问题")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

