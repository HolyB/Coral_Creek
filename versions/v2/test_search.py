
from services.search_service import get_search_service
import logging

logging.basicConfig(level=logging.INFO)

def test_search():
    s = get_search_service()
    
    # 测试1: A股 (东方财富)
    print("\n--- Test 1: 贵州茅台 (600519.SH) ---")
    res = s.get_stock_news("600519.SH", "贵州茅台")
    print(res[:500])
    
    # 测试2: 美股 (Yahoo)
    print("\n--- Test 2: AAPL ---")
    res = s.get_stock_news("AAPL", "Apple")
    print(res[:800])

if __name__ == "__main__":
    test_search()
