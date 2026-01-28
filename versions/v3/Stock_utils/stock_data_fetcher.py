from polygon import RESTClient
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import akshare as ak

class StockDataFetcher:
    def __init__(self, ticker: str, interval: str = '1d', start: str = None, end: str = None, source: str = 'yahoo'):
        """
        初始化 StockDataFetcher 类，设置股票代码、时间参数和数据源。

        :param ticker: 股票代码 (如 'AAPL')
        :param interval: 时间级别 ('1d', '1wk', '1mo', etc.)
        :param start: 起始时间 (格式为 'YYYY-MM-DD')，默认三年前
        :param end: 结束时间 (格式为 'YYYY-MM-DD')，默认当前时间
        :param source: 数据源 ('yahoo', 'alphavantage', 'polygon')
        """
        self.ticker = ticker
        self.interval = interval
        self.start = start
        self.end = end
        self.source = source

    def get_stock_data(self) -> pd.DataFrame:
        """
        根据数据源获取股票历史数据。

        :return: 包含历史数据的 DataFrame
        """
        # 设置默认起始和结束时间
        if self.start is None:
            self.start = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        if self.end is None:
            self.end = datetime.now().strftime('%Y-%m-%d')

        if self.source == 'yahoo':
            return self._get_data_from_yahoo()
        elif self.source == 'alphavantage':
            return self._get_data_from_alphavantage()
        elif self.source == 'polygon':
            return self._get_data_from_polygon()
        elif self.source == 'akshare':
            return self._get_data_from_akshare()
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    def _get_data_from_akshare(self) -> pd.DataFrame:
        """
        从 AKShare 获取A股数据。
        """
        try:
            # 处理股票代码格式
            if self.ticker.startswith('SH'):
                symbol = self.ticker[2:]  # 只需要数字部分
            elif self.ticker.startswith('SZ'):
                symbol = self.ticker[2:]
            else:
                symbol = self.ticker

            print(f"正在获取A股数据: {symbol}")

            # 获取日线数据，使用当前日期
            today = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')  # 获取一年数据
            
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=today,
                adjust="qfq"
            )

            print(f"获取数据时间范围: {start_date} 到 {today}")
            print(f"获取到的数据范围: {df['日期'].min()} 到 {df['日期'].max()}")

            # 重命名列以匹配统一格式
            df = df.rename(columns={
                '日期': 'Date',
                '开盘': 'Open',
                '最高': 'High',
                '最低': 'Low',
                '收盘': 'Close',
                '成交量': 'Volume'
            })

            # 设置日期索引
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # 确保数据类型正确
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

            return df

        except Exception as e:
            print(f"从 AKShare 获取数据失败 ({self.ticker}): {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


    def _get_data_from_yahoo(self) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取股票数据。

        :return: 包含历史数据的 DataFrame
        """
        df = yf.download(self.ticker, start=self.start, end=self.end, interval=self.interval)
        return df

    def _get_data_from_alphavantage(self) -> pd.DataFrame:
        """
        从 Alpha Vantage 获取股票数据。

        :return: 包含历史数据的 DataFrame
        """
        # 注意：需要安装 alpha_vantage 库，并获取 API 密钥
        from alpha_vantage.timeseries import TimeSeries
        api_key = 'YOUR_ALPHAVANTAGE_API_KEY'
        ts = TimeSeries(key=api_key, output_format='pandas')

        if self.interval == '1d':
            data, _ = ts.get_daily(symbol=self.ticker, outputsize='full')
        elif self.interval == '1wk':
            data, _ = ts.get_weekly(symbol=self.ticker)
        elif self.interval == '1mo':
            data, _ = ts.get_monthly(symbol=self.ticker)
        else:
            raise ValueError(f"Alpha Vantage 不支持的时间间隔: {self.interval}")

        # 过滤时间范围
        data = data[(data.index >= self.start) & (data.index <= self.end)]
        return data


    def _get_data_from_polygon(self) -> pd.DataFrame:
        """
        从 Polygon.io 获取股票数据。

        :return: 包含历史数据的 DataFrame
        """
        import requests

        api_key = 'qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D'  # 请替换为您的实际 API 密钥
        base_url = 'https://api.polygon.io'

        # 将 interval 映射到 multiplier 和 timespan
        interval_map = {
            '1m': (1, 'minute'),
            '5m': (5, 'minute'),
            '15m': (15, 'minute'),
            '30m': (30, 'minute'),
            '1h': (1, 'hour'),
            '2h': (2, 'hour'),
            '1d': (1, 'day'),
            '1wk': (1, 'week'),
            '1mo': (1, 'month')
        }

        if self.interval not in interval_map:
            raise ValueError(f"Polygon 不支持的时间间隔: {self.interval}")

        multiplier, timespan = interval_map[self.interval]

        # 将日期转换为 Unix 毫秒时间戳
        if isinstance(self.start, str):
            start_datetime = datetime.strptime(self.start, '%Y-%m-%d')
        else:
            start_datetime = self.start

        if isinstance(self.end, str):
            end_datetime = datetime.strptime(self.end, '%Y-%m-%d')
        else:
            end_datetime = self.end

        # 对于分钟和小时级别的数据，将时间设置为当天的开始和结束时间
        if timespan in ['minute', 'hour']:
            start_datetime = datetime.combine(start_datetime.date(), datetime.min.time())
            end_datetime = datetime.combine(end_datetime.date(), datetime.max.time())

        start_timestamp = int(start_datetime.timestamp() * 1000)
        end_timestamp = int(end_datetime.timestamp() * 1000)

        url = f"{base_url}/v2/aggs/ticker/{self.ticker.upper()}/range/{multiplier}/{timespan}/{start_timestamp}/{end_timestamp}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"

        # 调试信息
        print(f"Requesting URL: {url}")

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Polygon API 请求失败，状态码 {response.status_code}: {response.text}")

        data = response.json()
        if 'results' not in data:
            raise Exception(f"Polygon API 未返回数据: {data}")

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 重命名列以匹配 Yahoo Finance 数据格式
        df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume',
            'n': 'Transactions',
            'vw': 'VWAP'
        }, inplace=True)

        # 选择相关列
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        return df

# 示例调用
if __name__ == "__main__":
    # fetcher = StockDataFetcher('AAPL', source='yahoo')
    # df = fetcher.get_stock_data()
    # print(df.head())

    # fetcher_polygon = StockDataFetcher('AAPL', source='polygon')
    # df_polygon = fetcher_polygon.get_stock_data()
    # print(df_polygon.head())

    # 测试A股
    fetcher_cn = StockDataFetcher('SH600519', source='akshare')  # 茅台
    df_cn = fetcher_cn.get_stock_data()
    print("\nA股数据示例:")
    print(df_cn.head())