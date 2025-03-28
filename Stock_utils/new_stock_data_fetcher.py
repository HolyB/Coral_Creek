import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak
import yfinance as yf
import time
import os
import random
import requests

class NewStockDataFetcher:
    """
    股票数据获取类，支持不同数据源和周期
    """
    
    def __init__(self, symbol, source='akshare', interval='1d', lookback_days=365):
        """
        初始化数据获取器
        
        参数:
        symbol (str): 股票代码，格式视数据源而定
        source (str): 数据源 'akshare' 或 'yfinance'
        interval (str): 数据周期 '1d', '1h', '30m' 等
        lookback_days (int): 回溯天数
        """
        self.symbol = symbol
        self.source = source
        self.interval = interval
        
        # 设置时间范围
        self.end_date = datetime.now().strftime('%Y%m%d')
        self.start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y%m%d')
        
    def get_stock_data(self):
        """
        根据数据源获取股票数据，带缓存机制
        
        返回:
        pd.DataFrame: 股票OHLCV数据
        """
        # 尝试先从缓存获取数据
        df = self._get_data_from_cache()
        if df is not None:
            return df
            
        # 缓存不存在或已过期，从数据源获取
        if self.source.lower() == 'akshare':
            df = self._get_data_from_akshare(self.symbol)
        elif self.source.lower() == 'yfinance':
            df = self._get_data_from_yfinance(self.symbol)
        else:
            print(f"不支持的数据源: {self.source}")
            return None
            
        # 如果获取到数据，保存到缓存
        if df is not None and not df.empty:
            self._save_data_to_cache(df)
            
        return df
    
    def _get_data_from_cache(self, cache_days=1):
        """从缓存获取数据，如果缓存不存在或已过期则返回None"""
        try:
            cache_dir = "stock_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                return None
                
            cache_file = f"{cache_dir}/{self.symbol}.csv"
            
            # 检查缓存是否存在且未过期
            if os.path.exists(cache_file):
                file_time = os.path.getmtime(cache_file)
                if time.time() - file_time < cache_days * 86400:  # 缓存未过期
                    try:
                        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        # 简化输出，避免过多日志
                        return df
                    except Exception:
                        # 缓存读取失败，不打印错误，继续获取新数据
                        pass
            return None
        except Exception:
            # 检查缓存失败，不打印错误
            return None
    
    def _save_data_to_cache(self, df):
        """将数据保存到缓存"""
        try:
            cache_dir = "stock_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            cache_file = f"{cache_dir}/{self.symbol}.csv"
            df.to_csv(cache_file)
            # 不打印缓存成功信息，减少输出
        except Exception:
            # 缓存失败不影响主流程，也不打印错误
            pass
    
    def _get_data_from_akshare(self, symbol, max_retries=2):
        """从AKShare获取股票数据，处理各种类型的股票代码，带重试机制"""
        for retry in range(max_retries):
            try:
                # 最小的随机延迟，只在重试时等待更长时间
                if retry > 0:
                    time.sleep(random.uniform(0.5, 1.0))
                
                # 提取市场前缀和股票代码
                prefix = symbol[:2]
                code = symbol[2:]
                
                # 处理北交所股票
                if prefix == 'BJ':
                    try:
                        # 尝试使用北交所专用函数获取数据
                        if hasattr(ak, 'stock_bj_daily'):
                            df = ak.stock_bj_daily(
                                symbol=code,
                                start_date=self.start_date,
                                end_date=self.end_date,
                                adjust="qfq"
                            )
                        elif hasattr(ak, 'stock_bj_history_daily_em'):
                            df = ak.stock_bj_history_daily_em(
                                symbol=code,
                                start_date=self.start_date,
                                end_date=self.end_date,
                                adjust="qfq"
                            )
                        else:
                            # 尝试使用通用股票历史数据函数
                            df = ak.stock_zh_a_hist(
                                symbol=code, 
                                start_date=self.start_date, 
                                end_date=self.end_date, 
                                adjust="qfq"
                            )
                    except Exception as e:
                        # 北交所股票获取失败，打印错误并返回None
                        print(f"获取北交所股票 ({symbol}) 数据失败: {e}")
                        return None
                    
                # 科创板 (688开头)
                elif code.startswith('688'):
                    # 尝试使用stock_zh_a_daily函数获取科创板数据
                    try:
                        try:
                            df = ak.stock_zh_a_daily(
                                symbol=code, 
                                start_date=self.start_date,
                                end_date=self.end_date,
                                adjust="qfq"
                            )
                        except AttributeError:
                            # 如果stock_zh_a_daily不存在，尝试使用period参数的stock_zh_a_hist
                            df = ak.stock_zh_a_hist(
                                symbol=code, 
                                period="daily",
                                start_date=self.start_date, 
                                end_date=self.end_date, 
                                adjust="qfq"
                            )
                    except Exception:
                        # 如果仍然失败，尝试使用科创板特定函数
                        try:
                            # 检查是否有科创板专用函数
                            if hasattr(ak, 'stock_zh_kcb_daily'):
                                df = ak.stock_zh_kcb_daily(
                                    symbol=code,
                                    start_date=self.start_date,
                                    end_date=self.end_date,
                                    adjust="qfq"
                                )
                            else:
                                # 获取失败，直接返回None
                                return None
                        except Exception:
                            return None
                
                # 上海主板 (60开头)
                elif code.startswith('60'):
                    df = ak.stock_zh_a_hist(
                        symbol=code, 
                        start_date=self.start_date, 
                        end_date=self.end_date, 
                        adjust="qfq"  # 前复权
                    )
                    
                # 深圳创业板 (3开头)
                elif code.startswith('3'):
                    df = ak.stock_zh_a_hist(
                        symbol=code, 
                        start_date=self.start_date, 
                        end_date=self.end_date, 
                        adjust="qfq"
                    )
                        
                # 其他A股 (00开头的深市主板, 002开头的中小板等)
                else:
                    df = ak.stock_zh_a_hist(
                        symbol=code, 
                        start_date=self.start_date, 
                        end_date=self.end_date, 
                        adjust="qfq"  # 前复权
                    )
                
                # 检查数据是否为空
                if df is None or df.empty:
                    return None
                
                # 重命名列（根据实际返回的列名调整）
                column_mappings = {
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume',
                    '成交额': 'Amount',
                    '振幅': 'Amplitude',
                    '涨跌幅': 'Change',
                    '涨跌额': 'ChangeAmount',
                    '换手率': 'Turnover',
                    'open': 'Open',
                    'close': 'Close',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume'
                }
                
                # 仅重命名存在的列
                rename_dict = {k: v for k, v in column_mappings.items() if k in df.columns}
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                
                # 如果Date不在列中，但df有索引且索引是日期类型，将索引转为Date列
                if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={'index': 'Date'})
                
                # 设置索引
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                
                # 确保必要的列存在并是数值类型
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col not in df.columns:
                        return None
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 按日期排序
                df = df.sort_index()
                
                return df
                
            except requests.exceptions.ReadTimeout:
                # 处理请求超时
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # 逐次增加等待时间
                    time.sleep(wait_time)
                else:
                    return None
            except Exception as e:
                print(f"获取数据出错 ({symbol}): {e}")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 1.5
                    time.sleep(wait_time)
                else:
                    return None
        
        # 所有重试都失败
        return None
    
    def _get_data_from_yfinance(self, symbol):
        """从 Yahoo Finance 获取股票数据"""
        try:
            # 使用 yfinance 获取数据
            ticker = yf.Ticker(symbol)
            
            # 获取历史数据
            start_date_dt = datetime.strptime(self.start_date, '%Y%m%d')
            end_date_dt = datetime.strptime(self.end_date, '%Y%m%d')
            
            df = ticker.history(
                start=start_date_dt,
                end=end_date_dt + timedelta(days=1),
                interval=self.interval
            )
            
            # 检查数据是否为空
            if df is None or df.empty:
                return None
            
            # 确保必要的列存在并是数值类型
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    return None
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception:
            return None