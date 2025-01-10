import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time
import os
from polygon import RESTClient

from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.stock_analysis import StockAnalysis
from scan_signals_multi_thread_v2 import get_us_tickers

def calculate_features(df):
    """计算每日特征"""
    try:
        # 1. 基础价格和成交量变化
        df['price_change'] = df['Close'].pct_change()
        df['price_change_1d'] = df['price_change'].shift(-1)  # 第二天的涨跌幅（作为标签）
        df['volume_change'] = df['Volume'].pct_change()
        
        # 2. N日涨跌幅
        for n in [3, 5, 10, 20]:
            df[f'price_change_{n}d'] = df['Close'].pct_change(periods=n)
            df[f'volume_change_{n}d'] = df['Volume'].pct_change(periods=n)
        
        # 3. 计算移动平均
        for n in [5, 10, 20, 30]:
            df[f'ma_{n}'] = df['Close'].rolling(window=n).mean()
            df[f'volume_ma_{n}'] = df['Volume'].rolling(window=n).mean()
            df[f'price_vs_ma_{n}'] = df['Close'] / df[f'ma_{n}'] - 1
            df[f'volume_vs_ma_{n}'] = df['Volume'] / df[f'volume_ma_{n}'] - 1
        
        # 4. 波动率和振幅
        df['volatility'] = df['Close'].rolling(window=20).std()
        df['amplitude'] = (df['High'] - df['Low']) / df['Close'].shift(1)
        
        # 5. 添加 StockAnalysis 中的指标
        analysis = StockAnalysis(df)
        
        # 5.1 Phantom 指标
        df_phantom = analysis.calculate_phantom_indicators()
        phantom_cols = ['PINK', 'BLUE', '笑脸信号_做多', '笑脸信号_做空']
        for col in phantom_cols:
            df[f'daily_{col}'] = df_phantom[col]
        
        # 5.2 热力图和成交量指标
        df_volume = analysis.calculate_heatmap_volume()
        volume_cols = ['VOL_TIMES', 'HVOL_COLOR', 'GOLD_VOL', 'DOUBLE_VOL']
        for col in volume_cols:
            df[col] = df_volume[col]
        
        # 5.3 MACD信号
        df_macd = analysis.calculate_macd_signals()
        macd_cols = ['DIF', 'DEA', 'MACD', 'EMAMACD', 
                     '零轴下金叉', '零轴上金叉', '零轴上死叉', '零轴下死叉',
                     '先机信号', '底背离', '顶背离']
        for col in macd_cols:
            df[col] = df_macd[col]
        
        # 6. 计算BLUE信号持续天数
        df['BLUE_above_150'] = (df['daily_BLUE'] > 150).astype(int)
        df['BLUE_above_150_days'] = df['BLUE_above_150'].rolling(window=10).sum()
        
        # 7. 计算周线指标
        # 重采样为周线数据
        weekly = df.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        # 计算周线Phantom指标
        analysis_weekly = StockAnalysis(weekly)
        weekly_phantom = analysis_weekly.calculate_phantom_indicators()
        
        # 将周线数据重新映射到日线时间戳
        for col in phantom_cols:
            df[f'weekly_{col}'] = weekly_phantom[col].reindex(df.index, method='ffill')
        
        # 计算周线BLUE信号持续周数
        df['weekly_BLUE_above_150'] = (df['weekly_BLUE'] > 150).astype(int)
        df['weekly_BLUE_above_150_weeks'] = df['weekly_BLUE_above_150'].rolling(window=10).sum()
        
        return df
    
    except Exception as e:
        print(f"计算特征时出错: {e}")
        return None

def get_polygon_data(symbol, api_key):
    """从 Polygon 获取股票数据"""
    try:
        client = RESTClient(api_key)
        
        # 获取过去两年的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        aggs = []
        for a in client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            limit=50000
        ):
            aggs.append({
                'Date': pd.Timestamp.fromtimestamp(a.timestamp/1000),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume,
            })
            
        if not aggs:
            return None
            
        df = pd.DataFrame(aggs)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"获取Polygon数据失败: {e}")
        return None

def prepare_stock_data(symbol, name, api_key):
    """准备单个股票的数据"""
    try:
        # 从 Polygon 获取数据
        data = get_polygon_data(symbol, api_key)
        
        if data is None or data.empty:
            print(f"{symbol} {name} 无法获取数据")
            return None
            
        # 计算特征
        df = calculate_features(data)
        if df is None:
            return None
            
        # 添加股票信息
        df['symbol'] = symbol
        df['name'] = name
        
        # 删除包含 NaN 的行
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"{symbol} {name} 处理失败: {e}")
        return None

def main():
    """主函数 - 美股数据准备"""
    start_time = time.time()
    print("开始准备美股机器学习数据...")
    
    # 获取 Polygon API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("未找到 POLYGON_API_KEY 环境变量")
        return
    
    # 获取美股列表
    stock_list = get_us_tickers()
    
    if stock_list.empty:
        print("获取美股列表失败")
        return
    
    print(f"\n共获取到 {len(stock_list)} 只美股")
    
    # 存储所有股票的数据
    all_data = []
    failed_stocks = []
    
    # 处理每只股票
    with tqdm(total=len(stock_list), desc="处理进度") as pbar:
        for idx, stock in stock_list.iterrows():
            df = prepare_stock_data(stock['symbol'], stock['name'], api_key)
            if df is not None:
                all_data.append(df)
            else:
                failed_stocks.append((stock['symbol'], stock['name']))
            pbar.update(1)
            
            # 每处理100只股票保存一次中间结果
            if len(all_data) > 0 and len(all_data) % 100 == 0:
                interim_df = pd.concat(all_data, ignore_index=True)
                interim_df.to_csv(f'us_stock_ml_data_interim_{len(all_data)}.csv', 
                                index=True, encoding='utf-8-sig')
                print(f"\n已保存中间结果，当前处理了 {len(all_data)} 只股票")
    
    # 合并所有数据
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 保存数据
        final_df.to_csv('us_stock_ml_data.csv', index=True, encoding='utf-8-sig')
        print(f"\n数据已保存到 us_stock_ml_data.csv")
        print(f"总数据量: {len(final_df)} 行")
        print(f"成功处理股票数: {len(all_data)}")
        print(f"失败股票数: {len(failed_stocks)}")
        
        if failed_stocks:
            print("\n处理失败的股票:")
            for code, name in failed_stocks:
                print(f"- {code} {name}")
            
            # 保存失败列表
            with open('us_failed_stocks.txt', 'w', encoding='utf-8') as f:
                for code, name in failed_stocks:
                    f.write(f"{code},{name}\n")
            print("\n失败股票列表已保存到 us_failed_stocks.txt")
        
        # 显示特征列表
        print("\n特征列表:")
        features = sorted(final_df.columns.tolist())
        for col in features:
            print(f"- {col}")
            
        # 保存特征列表
        with open('us_feature_list.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(features))
        print("\n特征列表已保存到 us_feature_list.txt")
        
        # 显示数据统计信息
        print("\n数据统计信息:")
        print(final_df.describe())
        
    else:
        print("没有成功处理任何数据")
    
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 