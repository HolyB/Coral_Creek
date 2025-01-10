import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.stock_analysis import StockAnalysis
from scan_cn_signals import get_cn_tickers

def calculate_features(df):
    """计算每日特征"""
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

def prepare_stock_data(symbol, name):
    """准备单个股票的数据"""
    try:
        # 获取日线数据
        fetcher = StockDataFetcher(symbol, source='akshare', interval='1d')
        data = fetcher.get_stock_data()
        
        if data is None or data.empty:
            print(f"{symbol} {name} 无法获取数据")
            return None
            
        # 计算特征
        df = calculate_features(data)
        
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
    """主函数"""
    print("开始准备机器学习数据...")
    
    # 获取股票列表
    stock_list = get_cn_tickers()
    
    if stock_list.empty:
        print("获取股票列表失败")
        return
    
    # 存储所有股票的数据
    all_data = []
    
    # 处理每只股票
    total_stocks = len(stock_list)
    for idx, stock in stock_list.iterrows():
        print(f"\n处理 {stock['code']} {stock['name']}... ({idx+1}/{total_stocks})")
        df = prepare_stock_data(stock['code'], stock['name'])
        if df is not None:
            all_data.append(df)
    
    # 合并所有数据
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 保存数据
        final_df.to_csv('stock_ml_data.csv', index=True, encoding='utf-8-sig')
        print(f"\n数据已保存到 stock_ml_data.csv")
        print(f"总数据量: {len(final_df)} 行")
        
        # 显示特征列表
        print("\n特征列表:")
        features = sorted(final_df.columns.tolist())
        for col in features:
            print(f"- {col}")
            
        # 保存特征列表
        with open('feature_list.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(features))
        print("\n特征列表已保存到 feature_list.txt")
        
    else:
        print("没有成功处理任何数据")

if __name__ == "__main__":
    main() 