
import requests
import xml.etree.ElementTree as ET
import re

def test_google_rss():
    query = "贵州茅台 stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-CN"
    
    print(f"Testing URL: {url}")
    try:
        resp = requests.get(url, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            # print(resp.text[:500])
            items = re.findall(r'<item>(.*?)</item>', resp.text, re.DOTALL)
            print(f"Found {len(items)} items")
            for item in items[:3]:
                title = re.search(r'<title>(.*?)</title>', item)
                if title:
                    print(f"- {title.group(1)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_google_rss()
