import pandas as pd
import numpy as np
from datetime import datetime

class StockAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        初始化 StockAnalysis 类，输入包含 OHLCV 数据的 DataFrame。
        :param df: 包含股票历史数据的 DataFrame
        """
        self.df = df

    def calculate_dongfang_huohua(self):
        """
        计算东方红火箭指标并添加到 DataFrame 中。
        """
        # 计算VAR1到VAR9和VARA
        self.df['VAR1'] = self.df['High'].rolling(window=500).max().ewm(span=21).mean()
        self.df['VAR2'] = self.df['High'].rolling(window=250).max().ewm(span=21).mean()
        self.df['VAR3'] = self.df['High'].rolling(window=90).max().ewm(span=21).mean()
        self.df['VAR4'] = self.df['Low'].rolling(window=500).min().ewm(span=21).mean()
        self.df['VAR5'] = self.df['Low'].rolling(window=250).min().ewm(span=21).mean()
        self.df['VAR6'] = self.df['Low'].rolling(window=90).min().ewm(span=21).mean()
        self.df['VAR7'] = ((self.df['VAR4'] * 0.96 + self.df['VAR5'] * 0.96 + self.df['VAR6'] * 0.96 + self.df['VAR1'] * 0.558 + self.df['VAR2'] * 0.558 + self.df['VAR3'] * 0.558) / 6).ewm(span=21).mean()
        self.df['VAR8'] = ((self.df['VAR4'] * 1.25 + self.df['VAR5'] * 1.23 + self.df['VAR6'] * 1.2 + self.df['VAR1'] * 0.55 + self.df['VAR2'] * 0.55 + self.df['VAR3'] * 0.65) / 6).ewm(span=21).mean()
        self.df['VAR9'] = ((self.df['VAR4'] * 1.3 + self.df['VAR5'] * 1.3 + self.df['VAR6'] * 1.3 + self.df['VAR1'] * 0.68 + self.df['VAR2'] * 0.68 + self.df['VAR3'] * 0.68) / 6).ewm(span=21).mean()
        self.df['VARA'] = ((self.df['VAR7'] * 3 + self.df['VAR8'] * 2 + self.df['VAR9']) / 6 * 1.738).ewm(span=21).mean()

        # 计算VARB到VARF
        self.df['VARB'] = self.df['Low'].shift(1)
        self.df['VARC'] = self.df['Low'].sub(self.df['VARB']).abs().rolling(window=3).mean() / self.df[['Low', 'VARB']].apply(lambda x: max(x[0] - x[1], 0), axis=1).rolling(window=3).mean() * 100
        self.df['VARD'] = self.df.apply(lambda row: row['VARC'] * 10 if row['Close'] * 1.35 <= row['VARA'] else row['VARC'] / 10, axis=1).ewm(span=3).mean()
        self.df['VARE'] = self.df['Low'].rolling(window=30).min()
        self.df['VARF'] = self.df['VARD'].rolling(window=30).max()

        # 计算火焰山底
        self.df['火焰山底'] = self.df.apply(lambda row: ((row['VARD'] + row['VARF'] * 2) / 2) if row['Low'] <= row['VARE'] else 0, axis=1).ewm(span=3).mean() / 618


    def calculate_all_strategies(self) -> pd.DataFrame:
        """
        调用所有策略函数并生成包含所有特征的完整 DataFrame。
        :return: 添加了所有策略特征的 DataFrame
        """
        # self.calculate_ai_intraday_peak_trough()
        self.calculate_dongfang_huohua()

        self.df['VAR1'] = self.df['Close'].ewm(span=12).mean() - self.df['Close'].ewm(span=26).mean()
        self.df['VAR18'] = self.df['VAR1'].ewm(span=9).mean()
        self.df['VAR19'] = 2000 * (2 * (self.df['VAR1'] - self.df['VAR18'])) / self.df['Close'].ewm(span=30).mean()
        self.df['冲刺红柱'] = np.where(self.df['VAR19'] >= 0, self.df['VAR19'], np.nan)
        self.df['冲刺白柱'] = np.where(self.df['VAR19'] < 0, self.df['VAR19'], np.nan)
        self.df['天线'] = 50

        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

        return self.df


# 示例代码
if __name__ == "__main__":
    # 读取股票数据，例如 AAPL 股票数据
    df = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)
    # 创建 StockAnalysis 类的实例
    analysis = StockAnalysis(df)
    # 计算所有策略
    df = analysis.calculate_all_strategies()
    # 打印或保存最终的 DataFrame
    print(df.head())