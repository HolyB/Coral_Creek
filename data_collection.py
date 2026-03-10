import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import os

def get_stock_list():
    """获取沪深300成分股列表"""
    print("获取沪深300成分股列表...")
    csi300_stocks = ak.index_stock_cons_weight_csindex(symbol="000300")
    stock_list = csi300_stocks['成分券代码'].tolist()
    print(f"共有{len(stock_list)}只股票")
    return stock_list

def collect_stock_data(stock_list, start_date="20080101", end_date="20201231"):
    """获取每只股票的数据"""
    print("\n开始获取股票数据...")
    data_list = []
    failed_stocks = []

    for stock in tqdm(stock_list):
        try:
            stock_data = ak.stock_zh_a_hist(
                symbol=stock,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            stock_data['stock_id'] = stock
            data_list.append(stock_data)
            time.sleep(1)  # 添加1秒延时避免请求过快
        except Exception as e:
            print(f"获取 {stock} 数据失败: {str(e)}")
            failed_stocks.append(stock)
            continue
    
    return data_list, failed_stocks

def save_and_check_data(data_list, output_file="csi300_202401.csv"):
    """合并、检查并保存数据"""
    # 合并所有数据
    all_data = pd.concat(data_list, axis=0)
    
    # 检查数据
    print("\n数据概览:")
    print("数据维度:", all_data.shape)
    print("时间范围:", all_data['日期'].min(), "到", all_data['日期'].max())
    print("股票数量:", len(all_data['stock_id'].unique()))
    
    # 保存数据
    all_data.to_csv(output_file, index=False)
    print(f"\n数据已保存到 {output_file}")
    
    return all_data

def main():
    # 创建输出目录
    os.makedirs("data", exist_ok=True)
    
    # 获取股票列表
    stock_list = get_stock_list()
    
    # 收集数据
    data_list, failed_stocks = collect_stock_data(stock_list)
    
    # 处理失败情况
    if failed_stocks:
        print(f"\n获取失败的股票数量: {len(failed_stocks)}")
        print("失败的股票代码:", failed_stocks)
    
    # 保存数据
    output_file = "data/csi300_202401.csv"
    all_data = save_and_check_data(data_list, output_file)
    
    return all_data

if __name__ == "__main__":
    main()