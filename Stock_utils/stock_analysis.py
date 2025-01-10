import pandas as pd
import numpy as np
from datetime import datetime
from .MyTT import *

class StockAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        初始化 StockAnalysis 类，输入包含至少 'Open','High','Low','Close' 列的 DataFrame。
        :param df: 包含股票历史数据的 DataFrame
        """
        self.df = df

    def MA(self, series, n):
        return series.rolling(window=n).mean()

    def AVEDEV(self, series, n):
        sma = self.MA(series, n)
        return (series - sma).abs().rolling(window=n).mean()

    def REF(self, series, n):
        return series.shift(n)

    def ZIG(self, close, percent=3, depth=22):
        """
        简易ZigZag实现（近似富途ZIG(3,22)）:
        当相对pivot价格变化超过3%确认转折方向，并要求转折间隔至少depth个bar。
        此实现为近似逻辑，与富途实际逻辑可能有差异，需要根据实际需求微调。
        """
        length = len(close)
        if length == 0:
            return pd.Series(np.nan, index=close.index)

        zig = np.full(length, np.nan)
        threshold = percent / 100.0

        pivot_index = 0
        pivot_price = close.iloc[0]
        zig[0] = pivot_price
        mode = 0  # 0: 未定方向, 1:上升模式, -1:下降模式

        for i in range(1, length):
            change = (close.iloc[i] - pivot_price) / pivot_price
            bars_since_pivot = i - pivot_index

            if mode == 0:
                if change > threshold:
                    mode = 1
                    pivot_price = close.iloc[i]
                    pivot_index = i
                    zig[i] = pivot_price
                elif change < -threshold:
                    mode = -1
                    pivot_price = close.iloc[i]
                    pivot_index = i
                    zig[i] = pivot_price
            elif mode == 1:
                # 上升中，如果下跌超过threshold并且bars_since_pivot>=depth，则确认顶部
                if (change < -threshold) and (bars_since_pivot >= depth):
                    pivot_price = close.iloc[i]
                    pivot_index = i
                    zig[i] = pivot_price
                    mode = -1
                else:
                    # 创新高则更新pivot
                    if close.iloc[i] > pivot_price:
                        pivot_price = close.iloc[i]
                        pivot_index = i
                        zig[i] = pivot_price
            else:  # mode == -1下降模式
                if (change > threshold) and (bars_since_pivot >= depth):
                    pivot_price = close.iloc[i]
                    pivot_index = i
                    zig[i] = pivot_price
                    mode = 1
                else:
                    # 创新低则更新pivot
                    if close.iloc[i] < pivot_price:
                        pivot_price = close.iloc[i]
                        pivot_index = i
                        zig[i] = pivot_price

        return pd.Series(zig, index=close.index)

    def TROUGHBARS(self, param1=3, param2=16, param3=1, high=None, low=None, close=None):
        """
        简易TROUGHBARS(3,16,1)实现:
        返回距离最近波谷的bar数，当当前bar为波谷则为0。
        使用ZIG函数识别转折点，若当前转折点低于上一个转折点则为波谷。

        根据实际情况可能需进一步修正。
        """
        zig_line = self.ZIG(close, percent=param1, depth=param2)
        trough_bars = np.full(len(close), np.nan)
        pivots = zig_line.dropna()
        pivot_indexes = pivots.index

        if len(pivot_indexes) < 2:
            return pd.Series(trough_bars, index=close.index)

        last_trough_idx = -1
        # 判断波谷：当前转折价低于前一个转折价
        for i in range(1, len(pivot_indexes)):
            prev_idx = pivot_indexes[i-1]
            cur_idx = pivot_indexes[i]
            prev_price = zig_line.loc[prev_idx]
            cur_price = zig_line.loc[cur_idx]

            if cur_price < prev_price:
                # 波谷
                last_trough_idx = close.index.get_loc(cur_idx)
                trough_bars[last_trough_idx] = 0

        # 填充波谷间距离
        if last_trough_idx == -1:
            return pd.Series(trough_bars, index=close.index)

        for i in range(last_trough_idx+1, len(close)):
            if np.isnan(trough_bars[i]):
                trough_bars[i] = i - last_trough_idx
            else:
                # 遇到新波谷重置
                last_trough_idx = i
                trough_bars[i] = 0

        return pd.Series(trough_bars, index=close.index)

    def calculate_heima_wangzi(self):
        """计算黑马王子指标"""
        df = self.df
        
        # 计算VAR1: (HIGH+LOW+CLOSE)/3
        df['VAR1'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # 计算VAR1的14日移动平均和平均绝对偏差
        df['VAR1_MA14'] = df['VAR1'].rolling(window=14).mean()
        df['VAR1_AVEDEV'] = df['VAR1'].rolling(window=14).apply(lambda x: abs(x - x.mean()).mean())
        
        # 计算VAR2: (VAR1-MA(VAR1,14))/(0.015*AVEDEV(VAR1,14))
        df['VAR2'] = (df['VAR1'] - df['VAR1_MA14']) / (0.015 * df['VAR1_AVEDEV'])
        
        # 计算VAR3: 判断是否为3根K线内的谷底，且当日振幅大于4%
        df['AMPLITUDE'] = (df['High'] - df['Low']) / df['Low'] * 100  # 当日振幅
        df['IS_TROUGH'] = df['Low'].rolling(window=16, center=True).apply(
            lambda x: 1 if len(x) >= 3 and x.iloc[len(x)//2] == min(x) else 0
        )
        df['VAR3'] = np.where((df['IS_TROUGH'] == 1) & (df['AMPLITUDE'] > 4), 80, 0)
        
        # 计算VAR4: 使用ZigZag判断趋势反转
        def calculate_zigzag(prices, deviation=3):
            # 简化的ZigZag实现
            highs = prices.rolling(window=3, center=True).max()
            lows = prices.rolling(window=3, center=True).min()
            return np.where(prices == highs, 1, np.where(prices == lows, -1, 0))
        
        df['ZIG'] = calculate_zigzag(df['Close'])
        df['VAR4'] = np.where(
            (df['ZIG'].shift(1) > df['ZIG'].shift(2)) &
            (df['ZIG'].shift(2) <= df['ZIG'].shift(3)) &
            (df['ZIG'].shift(3) <= df['ZIG'].shift(4)),
            50, 0
        )
        
        # 计算掘底买点和黑马信号
        df['掘底买点'] = np.where((df['VAR2'] < -110) & (df['VAR3'] > 0), 1, 0)
        df['黑马信号'] = np.where((df['VAR2'] < -110) & (df['VAR4'] > 0), 1, 0)
        
        return df

    def calculate_ai_intraday_peak_trough(self):
        """
        计算 AI 分时顶底指标并添加到 DataFrame 中。
        """
        # 定义 GTY1 过滤条件
        GTY = 1250915
        current_date = pd.Timestamp('now', tz=self.df.index.tz)#datetime.now()
        self.df['GTY1'] = np.where(self.df.index < current_date, 1, np.nan)
        # print("GTY1")
        # 使用 pd.concat 一次性添加多列，避免碎片化警告
        new_columns = pd.DataFrame(index=self.df.index)
        

        # print(self.df['Close'])
        # print("0")
        # print(self.df['Low'].rolling(window=34).min())
        # print("1")
        # print(self.df['High'].rolling(window=34).max())
        # print("2")
        # print(((self.df['Close'] - self.df['Low'].rolling(window=34).min()) /
        #                      (self.df['High'].rolling(window=34).max() - self.df['Low'].rolling(window=34).min()) * 100).ewm(span=3).mean())
        # print("3")
        # print(self.df['GTY1'])
        # # 计算 趋势线
        # # new_columns['趋势线'] = ((self.df['Close'] - self.df['Low'].rolling(window=34).min()) /
        # #                      (self.df['High'].rolling(window=34).max() - self.df['Low'].rolling(window=34).min()) * 100).ewm(span=3).mean() * self.df['GTY1']
        # print("%%%%%%%%%%%%%%%%%%%%%%")

        # 计算 警戒线
        new_columns['警戒线'] = 95 * self.df['GTY1']

        # 计算 XIAO1，引用前一个值
        new_columns['XIAO1'] = ((self.df['Low'] + self.df['Open'] + self.df['Close'] + self.df['High']) / 4).shift(1) * self.df['GTY1']

        # 使用 SMA 计算 XIAO2
        new_columns['XIAO2'] = self.df['GTY1'] * (self.df['Low'] - new_columns['XIAO1']).abs().rolling(window=13).mean() / (
                        (self.df['Low'] - new_columns['XIAO1']).rolling(window=10).apply(lambda x: max(x.max(), 0)))

        # 计算 XIAO3 和 XIAO4
        new_columns['XIAO3'] = self.df['GTY1'] * new_columns['XIAO2'].ewm(span=10).mean()
        new_columns['XIAO4'] = self.df['GTY1'] * self.df['Low'].rolling(window=33).min()

        # 根据条件计算 XIAO5
        new_columns['XIAO5'] = self.df['GTY1'] * new_columns.apply(lambda row: row['XIAO3'] if self.df.loc[row.name, 'Low'] <= row['XIAO4'] else 0, axis=1).ewm(span=3).mean()


        # 计算 主力出没 和 洗盘
        new_columns['主力出没'] = np.where((new_columns['XIAO5'] > new_columns['XIAO5'].shift(1)) & (new_columns['XIAO5'] > 0), new_columns['XIAO5'], np.nan)
        new_columns['洗盘'] = np.where((new_columns['XIAO5'] < new_columns['XIAO5'].shift(1)) & (new_columns['XIAO5'] > 0), new_columns['XIAO5'], np.nan)

        # 计算 AI 红柱和绿柱
        new_columns['AI红柱'] = np.where(new_columns['XIAO5'] > new_columns['XIAO5'].shift(1), new_columns['XIAO5'], np.nan)
        new_columns['AI绿柱'] = np.where(new_columns['XIAO5'] < new_columns['XIAO5'].shift(1), new_columns['XIAO5'], np.nan)

        # 计算 MACD 相关值
        new_columns['DIFF'] = self.df['Close'].ewm(span=12).mean() - self.df['Close'].ewm(span=26).mean()
        new_columns['DEA'] = new_columns['DIFF'].ewm(span=9).mean()
        new_columns['MACD'] = (new_columns['DIFF'] - new_columns['DEA']) * 2

        # 修正底背离和顶背离逻辑
        A1 = (new_columns['DIFF'] > new_columns['DEA']) & (new_columns['DIFF'].shift(1) < new_columns['DEA'].shift(1))
        A2 = (new_columns['DEA'] > new_columns['DIFF']) & (new_columns['DEA'].shift(1) < new_columns['DIFF'].shift(1))
        new_columns['底背离'] = (self.df['Close'] > self.df['Close'].shift(A1.sum() + 1)) & (new_columns['DIFF'] > new_columns['DIFF'].shift(A1.sum() + 1)) & (new_columns['DIFF'] > new_columns['DEA'])
        new_columns['顶背离'] = (self.df['Close'] < self.df['Close'].shift(A2.sum() + 1)) & (new_columns['DIFF'] < new_columns['DIFF'].shift(A2.sum() + 1)) & (new_columns['DEA'] > new_columns['DIFF'])

        # 添加特别的底背离和顶背离列
        new_columns['brown_底背离'] = np.where(new_columns['底背离'], 57, 0)
        new_columns['white_底背离'] = np.where(new_columns['底背离'], 13, np.nan)
        new_columns['green_顶背离'] = np.where(new_columns['顶背离'], 88, np.nan)

        # 将新计算的列合并回原始 DataFrame
        self.df = pd.concat([self.df, new_columns], axis=1)

        # 删除重复的列名
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]




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

    def calculate_tandi_xunyao(self):
        """
        计算探底寻妖指标并添加到 DataFrame 中。
        """
        # 计算VAR2到VAR8
        self.df['VAR2'] = self.df['Low'].shift(1)
        self.df['VAR3'] = self.df['Low'].sub(self.df['VAR2']).abs().rolling(window=13).mean() / self.df[['Low', 'VAR2']].apply(lambda x: max(x[0] - x[1], 0), axis=1).rolling(window=13).mean() * 100
        self.df['VAR4'] = self.df.apply(lambda row: row['VAR3'] * 13 if row['Close'] >= self.df['Close'].shift(1)[row.name] * 1.2 else row['VAR3'] / 13, axis=1).ewm(span=13).mean()
        self.df['VAR5'] = self.df['Low'].rolling(window=34).min()
        self.df['VAR6'] = self.df['VAR4'].rolling(window=34).max()
        self.df['VAR7'] = np.where(self.df['Low'] == self.df['Low'].rolling(window=56).min(), 1, 0)
        self.df['VAR8'] = self.df.apply(lambda row: ((row['VAR4'] + row['VAR6'] * 2) / 2) if row['Low'] <= row['VAR5'] else 0, axis=1).ewm(span=3).mean() / 618 * self.df['VAR7']

        # 计算探底点
        self.df['AA'] = self.df['VAR8'] > self.df['VAR8'].shift(1)
        self.df['XG'] = (self.df['Low'] <= self.df['Low'].rolling(window=100).min().shift(3) * 1.01)
        self.df['XGA'] = self.df['AA'] & self.df['XG']
        self.df['XG1'] = self.df['XGA'] > self.df['XGA'].shift(1)
        self.df['探底点'] = self.df['XG1'] > self.df['XG1'].shift(1)

    def calculate_baoliang_xunniu(self):
        """
        计算爆量寻牛指标并添加到 DataFrame 中。
        """
        # 计算爆量寻牛指标相关变量
        M = 55
        N = 34
        LC = self.df['Close'].shift(1)
        self.df['RSI'] = ((self.df['Close'] - LC).clip(lower=0).rolling(window=3).mean() / (self.df['Close'] - LC).abs().rolling(window=3).mean()) * 100
        self.df['FF'] = self.df['Close'].ewm(span=3).mean()
        self.df['MA15'] = self.df['Close'].ewm(span=21).mean()
        self.df['VAR1'] = np.where((self.df.index.year >= 2038) & (self.df.index.month >= 1), 0, 1)
        self.df['VAR2'] = self.df['Low'].shift(1) * self.df['VAR1']
        self.df['VAR3'] = self.df['Low'].sub(self.df['VAR2']).abs().rolling(window=3).mean() / self.df[['Low', 'VAR2']].apply(lambda x: max(x[0] - x[1], 0), axis=1).rolling(window=3).mean() * 100 * self.df['VAR1']
        self.df['VAR4'] = self.df.apply(lambda row: row['VAR3'] * 10 if row['Close'] >= self.df['Close'].shift(1)[row.name] * 1.3 else row['VAR3'] / 10, axis=1).ewm(span=3).mean() * self.df['VAR1']
        self.df['VAR5'] = self.df['Low'].rolling(window=30).min() * self.df['VAR1']
        self.df['VAR6'] = self.df['VAR4'].rolling(window=30).max() * self.df['VAR1']
        self.df['VAR7'] = np.where(self.df['Close'].rolling(window=58).mean().notna(), 1, 0) * self.df['VAR1']
        self.df['VAR8'] = self.df.apply(lambda row: ((row['VAR4'] + row['VAR6'] * 2) / 2) if row['Low'] <= row['VAR5'] else 0, axis=1).ewm(span=3).mean() / 618 * self.df['VAR7']
        self.df['爆量'] = np.where(self.df['VAR8'] > 100, 100, self.df['VAR8']) * self.df['VAR1']

        # 计算趋势线
        self.df['RSV'] = (self.df['Close'] - self.df['Low'].rolling(window=N).min()) / (self.df['High'].rolling(window=N).max() - self.df['Low'].rolling(window=N).min()) * 100
        self.df['K'] = self.df['RSV'].rolling(window=3).mean()
        self.df['D'] = self.df['K'].rolling(window=3).mean()
        self.df['J'] = 3 * self.df['K'] - 2 * self.df['D']
        self.df['V11'] = 3 * ((self.df['Close'] - self.df['Low'].rolling(window=M).min()) / (self.df['High'].rolling(window=M).max() - self.df['Low'].rolling(window=M).min()) * 100).rolling(window=5).mean() - 2 * ((self.df['Close'] - self.df['Low'].rolling(window=M).min()) / (self.df['High'].rolling(window=M).max() - self.df['Low'].rolling(window=M).min()) * 100).rolling(window=5).mean().rolling(window=3).mean()
        # print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
        self.df['爆量寻牛趋势线'] = self.df['V11'].ewm(span=3).mean()

        # 标识爆量寻牛点
        GUP1 = 100 - 3 * ((self.df['Close'] - self.df['Low'].rolling(window=75).min()) / (self.df['High'].rolling(window=75).max() - self.df['Low'].rolling(window=75).min()) * 100).rolling(window=20).mean() + 2 * ((self.df['Close'] - self.df['Low'].rolling(window=75).min()) / (self.df['High'].rolling(window=75).max() - self.df['Low'].rolling(window=75).min()) * 100).rolling(window=20).mean().rolling(window=15).mean()
        GUP2 = 100 - 3 * ((self.df['Open'] - self.df['Low'].rolling(window=75).min()) / (self.df['High'].rolling(window=75).max() - self.df['Low'].rolling(window=75).min()) * 100).rolling(window=20).mean() + 2 * ((self.df['Open'] - self.df['Low'].rolling(window=75).min()) / (self.df['High'].rolling(window=75).max() - self.df['Low'].rolling(window=75).min()) * 100).rolling(window=20).mean().rolling(window=15).mean()
        GUP3 = (GUP1 < GUP2.shift(1)) & (self.df['Volume'] > self.df['Volume'].shift(1)) & (self.df['Close'] > self.df['Close'].shift(1))
        self.df['爆量寻牛点'] = GUP3 & (GUP3.rolling(window=30).sum() == 1)

    def calculate_buy_sell_energy(self):
        """
        计算买卖能量指标并添加到 DataFrame 中。
        """
        self.df['能量'] = np.sqrt(self.df['Volume']) * ((self.df['Close'] - (self.df['High'] + self.df['Low']) / 2) / ((self.df['High'] + self.df['Low']) / 2))
        self.df['平滑能量'] = self.df['能量'].ewm(span=16).mean()
        self.df['能量惯性'] = self.df['平滑能量'].ewm(span=16).mean()

        # 标识买卖信号点
        self.df['能量买卖信号'] = (self.df['能量惯性'] > 0) & (self.df['能量惯性'].shift(1) < 0)

    def calculate_main_retail_positions(self):  #还不一致
        """
        计算主力和散户持仓指标并添加到 DataFrame 中。
        """
        # 计算主力持仓
        self.df['JJZ'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['QJ0Z'] = self.df['Volume'] / np.where(self.df['High'] == self.df['Low'], 4, self.df['High'] - self.df['Low'])
        self.df['买1'] = self.df['QJ0Z'] * (np.minimum(self.df['Open'], self.df['Close']) - self.df['Low'])
        self.df['买2'] = self.df['QJ0Z'] * (self.df['JJZ'] - np.minimum(self.df['Close'], self.df['Open']))
        self.df['卖1'] = self.df['QJ0Z'] * (np.maximum(self.df['Close'], self.df['Open']) - self.df['JJZ'])
        self.df['卖2'] = self.df['QJ0Z'] * (self.df['High'] - np.maximum(self.df['Open'], self.df['Close']))
        self.df['DTZ'] = ((self.df['买1'] + self.df['买2']) - (self.df['卖1'] + self.df['卖2'])) / 10000
        self.df['主力持仓'] = self.df['DTZ'].rolling(window=66).sum()

        # 计算散户持仓
        self.df['JJ'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['QJ0'] = self.df['Volume'] / np.where(self.df['High'] == self.df['Low'], 4, self.df['High'] - self.df['Low'])
        self.df['TD1'] = self.df['QJ0'] * (np.minimum(self.df['Open'], self.df['Close']) - self.df['Low'])
        self.df['DD1'] = self.df['QJ0'] * (self.df['JJ'] - np.minimum(self.df['Close'], self.df['Open']))
        self.df['DD2'] = self.df['QJ0'] * (self.df['High'] - np.maximum(self.df['Open'], self.df['Close']))
        self.df['TD2'] = self.df['QJ0'] * (np.maximum(self.df['Close'], self.df['Open']) - self.df['JJ'])
        self.df['TD'] = self.df['TD1'] - self.df['TD2']
        self.df['DA'] = self.df['DD1'] - self.df['DD2']
        self.df['XD1'] = 1 - (self.df['TD1'] + self.df['DD1'])
        self.df['XD2'] = 1 - (self.df['TD2'] + self.df['DD2'])
        self.df['TZ'] = self.df['Volume'] / self.df['Volume'].rolling(window=60).sum() * 100
        self.df['DT'] = ((self.df['XD1'] - self.df['XD2']) / 10000) / self.df['TZ']
        self.df['散户持仓'] = self.df['DT'].rolling(window=22).sum()

    def calculate_market_funds_flow(self):
        """
        计算大盘资金进出指标并添加到 DataFrame 中。
        """
        # 计算 RSI1
        LC = self.df['Close'].shift(1)
        self.df['RSI1'] = (self.df['Close'] - LC).clip(lower=0).rolling(window=6).mean() / (self.df['Close'] - LC).abs().rolling(window=6).mean() * 100

        # 计算 AR
        self.df['AR'] = (self.df['High'] - self.df['Open']).rolling(window=26).sum() / (self.df['Open'] - self.df['Low']).rolling(window=26).sum() * 100

        # 标识弱势点
        self.df['弱'] = (self.df['RSI1'] > 85) * 30

        # 计算 VARB 和 VARC
        self.df['VARB'] = (self.df['Close'] - LC).clip(lower=0).rolling(window=7).mean() / (self.df['Close'] - LC).abs().rolling(window=7).mean() * 100
        self.df['VARC'] = (self.df['Close'] - LC).clip(lower=0).rolling(window=13).mean() / (self.df['Close'] - LC).abs().rolling(window=13).mean() * 100
        self.df['VARD'] = self.df.index.to_series().expanding().count()

        # 计算强弱分界
        self.df['强弱分界'] = ((self.df['VARB'] < 20) & (self.df['VARC'] < 25) & (self.df['VARD'] > 50) & (self.df['AR'] < 70)) * 30

        # 计算资金进出趋势
        self.df['资金进出趋势'] = ((self.df['Close'] - self.df['Close'].rolling(window=7).mean()) / self.df['Close'].rolling(window=7).mean() * 480).ewm(span=2).mean() * 5

        # 计算散户资金
        self.df['散户'] = ((self.df['Close'] - self.df['Close'].rolling(window=11).mean()) / self.df['Close'].rolling(window=11).mean() * 480).ewm(span=7).mean() * 5

        # 标识买卖点
        self.df['BT1'] = (self.df['RSI1'] < 25)
        self.df['BT2'] = (self.df['资金进出趋势'] < -10) & (self.df['资金进出趋势'] > self.df['散户'])
        self.df['BT'] = (self.df['BT1'] | self.df['BT2']).rolling(window=3).sum() >= 2

        # 添加黄钻、绿钻、大底和博弈等列
        self.df['黄钻'] = self.df['弱'] > 0
        self.df['绿钻'] = self.df['强弱分界'] > 0
        self.df['大底'] = self.df['BT']
        self.df['博弈'] = self.df['BT2']

    def calculate_bottom_surge(self):
        """
        计算底部扶摇直上指标并添加到 DataFrame 中。
        """
        self.df['VAR1'] = (2 * self.df['Close'] + self.df['High'] + self.df['Low']) / 4
        self.df['VAR2'] = self.df['Low'].rolling(window=34).min()
        self.df['VAR3'] = self.df['High'].rolling(window=34).max()
        self.df['VAR4'] = np.where((self.df['VAR2'] <= -150) & (self.df['VAR2'] > -200) & (self.df['VAR3'] <= -150) & (self.df['VAR3'] > -200), 10, 0)
        self.df['VAR6'] = ((self.df['Close'] - self.df['Low'].rolling(window=14).min()) / (self.df['High'].rolling(window=14).max() - self.df['Low'].rolling(window=14).min()) * 100).rolling(window=4).mean()
        self.df['VAR7'] = ((self.df['Close'] - self.df['Low'].rolling(window=15).min()) / (self.df['High'].rolling(window=15).max() - self.df['Low'].rolling(window=15).min()) * 100).rolling(window=4).mean()
        self.df['VAR8'] = np.where(self.df.index < self.df.index[70], -120, np.where((self.df['VAR3'] <= -200) & (self.df['VAR2'] <= -150), 15, self.df['VAR4']) - 120)
        self.df['VAR9'] = (self.df['VAR7'] - 50).rolling(window=3).mean() * 2 + (self.df['VAR6'] - 50).rolling(window=3).mean() / 2
        self.df['VAR10'] = (self.df['High'] + self.df['Low'] + self.df['Close'] * 2) / 4
        self.df['VAR11'] = self.df['VAR10'].ewm(span=10).mean()
        self.df['VAR12'] = self.df['VAR10'].rolling(window=10).std()
        self.df['VAR13'] = (self.df['VAR10'] - self.df['VAR11']) * 100 / self.df['VAR12']
        self.df['VAR14'] = self.df['VAR13'].ewm(span=5).mean()
        self.df['VAR15'] = self.df['VAR14'].ewm(span=10).mean() + 100 / 2 - 5
        self.df['VAR16'] = self.df['VAR15'].ewm(span=4).mean()
        self.df['AA'] = ((self.df['VAR1'] - self.df['VAR2']) / (self.df['VAR3'] - self.df['VAR2']) * 100).ewm(span=13).mean()
        self.df['BB'] = (0.667 * self.df['AA'].shift(1) + 0.333 * self.df['AA']).ewm(span=2).mean()
        self.df['底部扶摇直上'] = (self.df['AA'] > self.df['BB']) & (self.df['AA'] < 20) & (self.df['AA'].shift(1) < self.df['BB'].shift(1))
        self.df['XG'] = (self.df['AA'] > self.df['BB']) & (self.df['AA'] < 20) & (self.df['AA'].shift(1) < self.df['BB'].shift(1))
        self.df['黄柱'] = (self.df['AA'] > 22) & (self.df['BB'] < self.df['AA'])
        self.df['逃顶'] = (self.df['BB'].shift(1) > self.df['AA'].shift(1)) & (self.df['AA'] > 80)
        self.df['是底'] = ((self.df['VAR15'] > self.df['VAR16']) & (self.df['VAR16'] < -10)) & (self.df['VAR9'] > self.df['VAR8'])

    def calculate_start_uptrend(self):
        """
        计算启动上涨指标并添加到 DataFrame 中。
        """
        self.df['XX'] = self.df['Close'].sub(self.df['Open']).mul(2).add(self.df['Open']).ewm(span=20).mean().ewm(span=10).mean()
        self.df['YY'] = self.df['Close'].sub(self.df['Open']).mul(2).add(self.df['Open']).ewm(span=3).mean().ewm(span=10).mean()
        self.df['趋势多'] = (self.df['YY'] > self.df['XX']) & (self.df['YY'].shift(1) <= self.df['XX'].shift(1))
        self.df['趋势空'] = (self.df['XX'] > self.df['YY']) & (self.df['XX'].shift(1) <= self.df['YY'].shift(1))
        self.df['短线看红涨绿跌'] = self.df['Close'].rolling(window=5).mean().ewm(span=3).mean()
        self.df['买线'] = self.df['Close'].ewm(span=5).mean()
        self.df['卖线'] = self.df['Close'].rolling(window=21).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] * 20 + x[-1], raw=True).ewm(span=42).mean()
        self.df['BU'] = (self.df['买线'] > self.df['卖线']) & (self.df['买线'].shift(1) <= self.df['卖线'].shift(1))
        self.df['SEL'] = (self.df['卖线'] > self.df['买线']) & (self.df['卖线'].shift(1) <= self.df['买线'].shift(1))
        self.df['启动上涨'] = (self.df['Close'].rolling(window=3).mean() > self.df['Close'].rolling(window=6).mean()) & (self.df['Close'] > self.df['Close'].shift(1) * 1.05)

    def calculate_runway_sprint(self): #白柱可能有点问题
        """
        计算跑道冲刺指标并添加到 DataFrame 中。
        """
        self.df['VAR1'] = self.df['Close'].ewm(span=12).mean() - self.df['Close'].ewm(span=26).mean()
        self.df['VAR18'] = self.df['VAR1'].ewm(span=9).mean()
        self.df['VAR19'] = 2000 * (2 * (self.df['VAR1'] - self.df['VAR18'])) / self.df['Close'].ewm(span=30).mean()
        self.df['冲刺红柱'] = np.where(self.df['VAR19'] >= 0, self.df['VAR19'], np.nan)
        self.df['下降白柱'] = np.where(self.df['VAR19'] < 0, self.df['VAR19'], np.nan)
        self.df['天线'] = 50

    def calculate_quantitative_macd(self):
        """
        计算量化 MACD 指标并添加到 DataFrame 中。
        """
        self.df['DIFF'] = self.df['Close'].ewm(span=12).mean() - self.df['Close'].ewm(span=26).mean()
        self.df['DEA'] = self.df['DIFF'].ewm(span=9).mean()
        self.df['MACD'] = 2 * (self.df['DIFF'] - self.df['DEA'])
        self.df['柱1'] = np.where(self.df['DIFF'] > self.df['DEA'], self.df['DIFF'], 0)
        self.df['柱2'] = np.where(self.df['DEA'] < self.df['DIFF'], self.df['DEA'], 0)
        self.df['VA'] = np.where(self.df['Close'] > self.df['Close'].shift(1), self.df['Volume'], -self.df['Volume'])
        self.df['OBV1'] = self.df['VA'].where(self.df['Close'] != self.df['Close'].shift(1), 0).cumsum()
        self.df['OBV2'] = self.df['OBV1'].ewm(span=3).mean() - self.df['OBV1'].rolling(window=9).mean()
        self.df['OBV3'] = self.df['OBV2'].where(self.df['OBV2'] > 0, 0).ewm(span=3).mean()
        self.df['MAC3'] = self.df['Close'].rolling(window=3).mean()
        self.df['红箭头'] = (self.df['DIFF'] > self.df['DEA']) & (self.df['DIFF'].shift(1) <= self.df['DEA'].shift(1))
        self.df['绿箭头'] = (self.df['DEA'] > self.df['DIFF']) & (self.df['DEA'].shift(1) <= self.df['DIFF'].shift(1))

    def calculate_label_statistics(self):
        """
        计算前5天、前15天、前30天和前90天内的指标信号次数和最近一次信号对应价格。
        """
        signals = [
            'AI红柱', 'AI绿柱', 'brown_底背离', 'white_底背离', 'green_顶背离',
            '火焰山底', '探底点', '爆量寻牛点', '能量买卖信号', '黄钻', '绿钻',
            '大底', '底部扶摇直上', 'BU', 'SEL', '启动上涨', '红箭头', '绿箭头', '趋势空', '启动上涨','逃顶'
        ]
        periods = [5, 15, 30, 90]

        new_columns = {}

        for signal in signals:
            if signal in self.df.columns:
                for period in periods:
                    count_col = f'{signal}_count_last_{period}d'
                    price_col = f'{signal}_last_price_{period}d'

                    # Calculate the count of signals in the given period
                    new_columns[count_col] = self.df[signal].rolling(window=period, min_periods=1).apply(
                        lambda x: np.nansum(~np.isnan(x)), raw=True
                    )

                    # Calculate the price corresponding to the last signal in the given period
                    def get_last_price(x):
                        last_valid_idx = np.where(~np.isnan(x))[0]
                        if len(last_valid_idx) > 0:
                            # 使用 x.index[last_valid_idx[-1]] 获取的原始索引可能是问题的来源
                            # 我们可以通过相对位置来找回相应价格
                            return self.df['Close'].iloc[-(len(x) - last_valid_idx[-1])]
                        return np.nan

                    new_columns[price_col] = self.df[signal].rolling(window=period, min_periods=1).apply(
                        get_last_price, raw=True
                    )

        # 将新计算的列一次性添加回原始 DataFrame 中，避免逐列插入导致的碎片化问题
        new_columns_df = pd.DataFrame(new_columns, index=self.df.index)
        self.df = pd.concat([self.df, new_columns_df], axis=1)

    def calculate_phantom_indicators(self):
        """计算幻影主力指标"""
        try:
            df = self.df
            
            # 海底捞月部分
            df['VAR1'] = (df['Low'] + df['Open'] + df['Close'] + df['High']).shift(1) / 4
            
            # 修正 VAR2 计算
            abs_diff = (df['Low'] - df['VAR1']).abs()
            max_diff = np.maximum(df['Low'] - df['VAR1'], 0)
            df['VAR2'] = abs_diff.rolling(window=13, min_periods=1).mean() / \
                        max_diff.rolling(window=10, min_periods=1).mean()
            
            # 修正 VAR3 计算
            df['VAR3'] = df['VAR2'].ewm(span=10, adjust=False, min_periods=1).mean()
            
            # 修正 VAR4, VAR5, VAR6 计算
            df['VAR4'] = df['Low'].rolling(window=33, min_periods=1).min()
            df['VAR5'] = np.where(df['Low'] <= df['VAR4'], df['VAR3'], 0)
            df['VAR5'] = pd.Series(df['VAR5']).ewm(span=3, adjust=False, min_periods=1).mean()
            df['VAR6'] = np.where(df['VAR5'] > 0, np.power(df['VAR5'], 0.3), 0)
            
            # 负向计算
            abs_diff_high = (df['High'] - df['VAR1']).abs()
            min_diff_high = np.minimum(df['High'] - df['VAR1'], 0)
            df['VAR21'] = abs_diff_high.rolling(window=13, min_periods=1).mean() / \
                        min_diff_high.rolling(window=10, min_periods=1).mean()
            df['VAR31'] = df['VAR21'].ewm(span=10, adjust=False, min_periods=1).mean()
            df['VAR41'] = df['High'].rolling(window=33, min_periods=1).max()
            df['VAR51'] = np.where(df['High'] >= df['VAR41'], -df['VAR31'], 0)
            df['VAR51'] = pd.Series(df['VAR51']).ewm(span=3, adjust=False, min_periods=1).mean()
            df['VAR61'] = np.where(df['VAR51'] < 0, np.power(-df['VAR51'], 0.3), 0)
            
            # 修正缩放比例计算
            max_var = max(df['VAR6'].max(), df['VAR61'].max())
            if max_var > 0:
                RADIO1 = 200 / max_var
            else:
                RADIO1 = 200
            
            # 修正 BLUE 和 LIRED 计算
            df['BLUE'] = np.where(df['VAR5'] > df['VAR5'].shift(1), df['VAR6'] * RADIO1, 0)
            df['LIRED'] = np.where(df['VAR51'] > df['VAR51'].shift(1), -df['VAR61'] * RADIO1, 0)
            
            # 资金力度部分
            df['QJJ'] = df['Volume'] / ((df['High'] - df['Low']) * 2 - (df['Close'] - df['Open']).abs())
            df['XVL'] = np.where(df['Close'] == df['Open'], 0, (df['Close'] - df['Open']) * df['QJJ'])
            df['HSL'] = df['XVL'] / 20 / 1.15
            
            # 攻击流量
            df['攻击流量'] = df['HSL'] * 0.55 + df['HSL'].shift(1) * 0.33 + df['HSL'].shift(2) * 0.22
            df['LLJX'] = pd.Series(df['攻击流量']).ewm(span=3, adjust=False, min_periods=1).mean()
            
            # 缩放比例
            RADIO = 10000 / df['Volume'].max() if df['Volume'].max() > 0 else 10000
            
            # 计算资金指标
            df['RED'] = np.where(df['LLJX'] > 0, df['LLJX'] * RADIO, 0)
            df['YELLOW'] = np.where(df['HSL'] > 0, df['HSL'] * 0.6 * RADIO, 0)
            df['GREEN'] = np.where((df['LLJX'] < 0) | (df['HSL'] < 0), 
                                np.minimum(df['LLJX'], df['HSL'] * 0.6) * RADIO, 0)
            
            # KDJ部分
            df['RSV1'] = (df['Close'] - df['Low'].rolling(39, min_periods=1).min()) / \
                        (df['High'].rolling(39, min_periods=1).max() - df['Low'].rolling(39, min_periods=1).min()) * 100
            df['K'] = df['RSV1'].rolling(2, min_periods=1).mean()
            df['D'] = df['K'].rolling(2, min_periods=1).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            df['PINK'] = pd.Series(df['J']).rolling(2, min_periods=1).mean()
            
            # LIGHTBLUE指标
            df['LIGHTBLUE'] = df['攻击流量'] * 0.222228 * RADIO
            
            # 笑脸信号
            df['笑脸信号_做多'] = np.where((df['PINK'].shift(1) < 10) & (df['PINK'] > 10), 1, 0)
            df['笑脸信号_做空'] = np.where((df['PINK'].shift(1) > 94) & (df['PINK'] < 94), 1, 0)
            
            self.df = df
            return self.df
            
        except Exception as e:
            print(f"计算幻影主力指标时出错: {str(e)}")
            return self.df

    def calculate_heatmap_volume(self):
        """计算热力图成交量指标"""
        if 'HVOL_COLOR' in self.df.columns:
            return
        
        # 参数设置
        LENGTH = 610
        SLENGTH = 610
        THRESEXTRAHIGH = 4
        THRESHIGH = 2.5
        THRESMEDIUM = 1
        THRESNORMAL = -0.5
        
        # 计算基础指标
        barindex = min(len(self.df), 1000)
        LEN = min(barindex + 1, LENGTH)
        SLEN = min(barindex + 1, SLENGTH)
        
        # 计算均值和标准差
        self.df['HVOL_MEAN'] = self.df['Volume'].rolling(LEN).mean()
        self.df['HVOL_DEV'] = self.df['Volume'].rolling(SLEN).std()
        
        # 计算偏离度
        self.df['HVOL_DEVBAR'] = (self.df['Volume'] - self.df['HVOL_MEAN']) / self.df['HVOL_DEV']
        
        # 计算热力图颜色级别
        conditions = [
            self.df['HVOL_DEVBAR'] >= THRESEXTRAHIGH,
            (self.df['HVOL_DEVBAR'] < THRESEXTRAHIGH) & (self.df['HVOL_DEVBAR'] >= THRESHIGH),
            (self.df['HVOL_DEVBAR'] < THRESHIGH) & (self.df['HVOL_DEVBAR'] >= THRESMEDIUM),
            (self.df['HVOL_DEVBAR'] < THRESMEDIUM) & (self.df['HVOL_DEVBAR'] >= THRESNORMAL),
            self.df['HVOL_DEVBAR'] < THRESNORMAL
        ]
        choices = [5, 4, 3, 2, 1]  # 5=深蓝, 4=蓝, 3=浅蓝, 2=灰, 1=浅灰
        self.df['HVOL_COLOR'] = np.select(conditions, choices, default=1)
        
        # 计算黄金柱和倍量柱
        self.df['VAR1'] = (3 * self.df['Close'] + self.df['High'] + self.df['Low'] + 2 * self.df['Open']) / 7
        self.df['VAR2'] = self.df['VAR1'].rolling(8).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
        
        # 条件判断
        self.df['VOL_MAX_10'] = self.df['Volume'] == self.df['Volume'].rolling(10).max()
        self.df['VOL_TIMES'] = self.df['Volume'] / self.df['Volume'].shift(1)
        self.df['VOL_2TIMES'] = self.df['VOL_TIMES'] > 2
        self.df['PRICE_COND'] = (self.df['Close'] > self.df['VAR2']) & (self.df['Close'] > self.df['Open'])
        
        # 黄金柱：10日内最大成交量 + 倍量 + 收阳且收盘价高于WMA8
        self.df['GOLD_VOL'] = self.df['VOL_MAX_10'] & self.df['VOL_2TIMES'] & self.df['PRICE_COND']
        
        # 倍量柱：成交量是前一日2倍以上
        self.df['DOUBLE_VOL'] = self.df['VOL_2TIMES']
        
        # 清理中间变量
        self.df.drop(['VAR1', 'VAR2', 'VOL_MAX_10', 'VOL_2TIMES', 'PRICE_COND'], axis=1, inplace=True)

    def calculate_divergence_signals(self, indicators):
        """计算顶底背离信号"""
        try:
            N11, N111 = 6, 3
            
            # 确保K1已经计算
            if 'K1' not in indicators.columns:
                print("错误：需要先计算KDJ指标")
                return
            
            # 初始化所有需要的列
            for col in ['A11', 'B11', 'C11', 'PERIOD_TOP11', 'TOP_DIV',
                       'A111', 'B111', 'C111', 'PERIOD_TOP', 'TOP_DIV111']:
                indicators[col] = pd.Series(False, index=self.df.index)
            
            # 计算N11周期顶背离
            def find_tops(k_series, n_periods):
                tops = pd.Series(False, index=k_series.index)
                for i in range(n_periods, len(k_series)):
                    if i >= 2*n_periods+1:
                        window = k_series.iloc[i-2*n_periods-1:i+1]
                        if k_series.iloc[i-n_periods] == window.max():
                            tops.iloc[i-n_periods] = True
                return tops
            
            def backset(series, n):
                result = series.copy()
                for i in range(len(series)):
                    if series.iloc[i]:
                        end_idx = min(i+n+1, len(series))
                        result.iloc[i:end_idx] = True
                return result
            
            def filter_signal(series, n):
                result = series.copy()
                for i in range(len(series)-n+1):
                    if i+n <= len(series):
                        result.iloc[i+n-1] = series.iloc[i:i+n].all()
                return result
            
            # N11周期顶背离
            indicators['A11'] = find_tops(indicators['K1'], N11)
            indicators['B11'] = backset(indicators['A11'], N11+1)
            indicators['C11'] = filter_signal(indicators['B11'], N11)
            
            # 计算PERIOD_TOP11
            last_true_idx = -1
            for i in range(len(indicators)):
                if indicators['C11'].iloc[i]:
                    last_true_idx = i
                if last_true_idx >= 0:
                    indicators['PERIOD_TOP11'].iloc[i] = i - last_true_idx
            
            # 计算顶背离信号
            for i in range(len(self.df)):
                period = int(indicators['PERIOD_TOP11'].iloc[i]) if not np.isnan(indicators['PERIOD_TOP11'].iloc[i]) else 0
                if i >= period > 0:
                    indicators['TOP_DIV'].iloc[i] = (
                        self.df['Close'].iloc[i-period] < self.df['Close'].iloc[i] and
                        indicators['K1'].iloc[i-period] > indicators['K1'].iloc[i] and
                        indicators['C11'].iloc[i]
                    )
            
            # N111周期顶背离（类似计算）
            indicators['A111'] = find_tops(indicators['K1'], N111)
            indicators['B111'] = backset(indicators['A111'], N111+1)
            indicators['C111'] = filter_signal(indicators['B111'], N111)
            
            # 计算PERIOD_TOP
            last_true_idx = -1
            for i in range(len(indicators)):
                if indicators['C111'].iloc[i]:
                    last_true_idx = i
                if last_true_idx >= 0:
                    indicators['PERIOD_TOP'].iloc[i] = i - last_true_idx
            
            # 计算N111顶背离信号
            for i in range(len(self.df)):
                period = int(indicators['PERIOD_TOP'].iloc[i]) if not np.isnan(indicators['PERIOD_TOP'].iloc[i]) else 0
                if i >= period > 0:
                    indicators['TOP_DIV111'].iloc[i] = (
                        self.df['High'].iloc[i-period] < self.df['High'].iloc[i] and
                        indicators['K1'].iloc[i-period] > indicators['K1'].iloc[i] and
                        indicators['C111'].iloc[i]
                    )
                
        except Exception as e:
            print(f"背离信号计算出错: {e}")
            raise

    def calculate_mtm_signals(self):
        """计算MTM相关信号"""
        # MTM基础计算
        self.df['MTM'] = self.df['Close'] - self.df['Close'].shift(1)
        
        # 其他指标计算
        self.df['VAR10'] = (self.df['Low'] + self.df['Open'] + 
                            self.df['Close'] + self.df['High']).shift(1) / 4
        
        abs_diff = (self.df['Low'] - self.df['VAR10']).abs()
        max_diff = (self.df['Low'] - self.df['VAR10']).clip(lower=0)
        
        self.df['VAR20'] = abs_diff.rolling(13).mean() / max_diff.rolling(10).mean()
        self.df['VAR30'] = self.df['VAR20'].ewm(span=10).mean()
        self.df['VAR40'] = self.df['Low'].rolling(33).min()
        
        # 主力进场和洗盘信号的计算
        cond = self.df['Low'] <= self.df['VAR40']
        self.df['VAR50'] = np.where(cond, self.df['VAR30'], 0).ewm(span=3).mean()
        
        # 计算高点相关指标
        abs_diff_high = (self.df['High'] - self.df['VAR10']).abs()
        min_diff_high = (self.df['High'] - self.df['VAR10']).clip(upper=0)
        
        self.df['VAR210'] = abs_diff_high.rolling(13).mean() / min_diff_high.abs().rolling(10).mean()
        self.df['VAR310'] = self.df['VAR210'].ewm(span=10).mean()
        self.df['VAR410'] = self.df['High'].rolling(33).max()
        
        cond_high = self.df['High'] >= self.df['VAR410']
        self.df['VAR510'] = np.where(cond_high, self.df['VAR310'], 0).ewm(span=3).mean()

    def calculate_all_indicators(self):
        """计算所有技术指标"""
        # 创建一个新的DataFrame来存储所有指标
        indicators = pd.DataFrame(index=self.df.index)
        
        # 计算各种指标
        self.calculate_kdj(indicators)
        self.calculate_var_signals(indicators)
        self.calculate_force_signals(indicators)
        self.calculate_divergence_signals(indicators)
        
        # 一次性合并所有指标
        self.df = pd.concat([self.df, indicators], axis=1)
        
        # 计算其他原有的指标
        self.calculate_ai_intraday_peak_trough()
        self.calculate_dongfang_huohua()
        self.calculate_tandi_xunyao()
        self.calculate_baoliang_xunniu()
        self.calculate_buy_sell_energy()
        self.calculate_main_retail_positions()
        self.calculate_market_funds_flow()
        self.calculate_bottom_surge()
        self.calculate_start_uptrend()
        self.calculate_runway_sprint()
        self.calculate_quantitative_macd()
        self.calculate_label_statistics()
        self.calculate_phantom_indicators()
        self.calculate_heatmap_volume()
        self.calculate_heima_wangzi()
        self.calculate_phantom_force()
        
        return self.df

    def calculate_var_signals(self, indicators):
        """计算VAR相关信号（包含掘底买点和黑马信号）"""
        # 计算VAR1-VAR4
        indicators['VAR1'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        indicators['VAR1_MA14'] = indicators['VAR1'].rolling(window=14).mean()
        indicators['VAR1_AVEDEV14'] = indicators['VAR1'].rolling(window=14).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        indicators['VAR2'] = (indicators['VAR1'] - indicators['VAR1_MA14']) / \
                            (0.015 * indicators['VAR1_AVEDEV14'])
        
        # 初始化VAR3和VAR4
        indicators['VAR3'] = 0
        indicators['VAR4'] = 0
        
        # 计算VAR3（掘底信号）
        def is_trough(series, lookback=16):
            if len(series) < lookback:
                return False
            mid = len(series) // 2
            return all(series[mid] <= series[i] for i in range(len(series)) if i != mid)
        
        for i in range(16, len(self.df)):
            price_window = self.df['Low'].iloc[i-16:i+1]
            if is_trough(price_window.values) and \
               self.df['High'].iloc[i] > self.df['Low'].iloc[i] + 0.04:
                indicators.loc[self.df.index[i], 'VAR3'] = 80
        
        # 计算VAR4（黑马信号）
        def calculate_zig(series, threshold=0.03):
            result = pd.Series(0.0, index=series.index)  # 初始化为float类型
            last_extreme = float(series.iloc[0])  # 确保是float类型
            last_extreme_idx = 0
            trend = 1  # 1 for up, -1 for down
            
            for i in range(1, len(series)):
                change = (float(series.iloc[i]) - last_extreme) / last_extreme
                if (trend == 1 and change < -threshold) or \
                   (trend == -1 and change > threshold):
                    result.iloc[last_extreme_idx] = last_extreme
                    last_extreme = float(series.iloc[i])
                    last_extreme_idx = i
                    trend *= -1
                    
            return result
        
        zig = calculate_zig(self.df['Close'])
        for i in range(3, len(zig)):
            if zig.iloc[i] > zig.iloc[i-1] and \
               zig.iloc[i-1] <= zig.iloc[i-2] and \
               zig.iloc[i-2] <= zig.iloc[i-3]:
                indicators.loc[zig.index[i], 'VAR4'] = 50
        
        # 计算最终信号
        indicators['掘底买点'] = (indicators['VAR2'] < -110) & (indicators['VAR3'] > 0)
        indicators['黑马信号'] = (indicators['VAR2'] < -110) & (indicators['VAR4'] > 0)

    def calculate_force_signals(self, indicators):
        """计算主力进场和洗盘信号"""
        # 计算基础变量
        indicators['VAR10'] = (self.df[['Low', 'Open', 'Close', 'High']].mean(axis=1)).shift(1)
        
        # 计算主力进场信号
        abs_diff = (self.df['Low'] - indicators['VAR10']).abs()
        max_diff = (self.df['Low'] - indicators['VAR10']).clip(lower=0)
        
        indicators['VAR20'] = abs_diff.rolling(13).mean() / max_diff.rolling(10).mean()
        indicators['VAR30'] = pd.Series(indicators['VAR20']).ewm(span=10).mean()  # 确保是Series
        indicators['VAR40'] = self.df['Low'].rolling(33).min()
        
        cond = self.df['Low'] <= indicators['VAR40']
        var30_series = pd.Series(indicators['VAR30'])  # 转换为Series
        indicators['VAR50'] = pd.Series(np.where(cond, var30_series, 0)).ewm(span=3).mean()
        
        # 计算最终信号
        indicators['主力进场'] = np.where(
            indicators['VAR50'] > indicators['VAR50'].shift(1),
            indicators['VAR50'],
            0
        )
        
        indicators['洗盘'] = np.where(
            indicators['VAR50'] < indicators['VAR50'].shift(1),
            indicators['VAR50'],
            0
        )

    def calculate_kdj(self, indicators):
        """计算KDJ相关信号"""
        try:
            N, M1, M2 = 9, 3, 3
            
            # 计算基础KDJ指标
            low_n = self.df['Low'].rolling(window=N).min()
            high_n = self.df['High'].rolling(window=N).max()
            
            # 初始化所有需要的列
            indicators['RSV'] = pd.Series(0.0, index=self.df.index)
            indicators['K'] = pd.Series(0.0, index=self.df.index)
            indicators['D'] = pd.Series(0.0, index=self.df.index)
            indicators['K1'] = pd.Series(0.0, index=self.df.index)
            
            # 计算RSV，处理除零情况
            denominator = (high_n - low_n)
            indicators['RSV'] = np.where(
                denominator != 0,
                (self.df['Close'] - low_n) / denominator * 100,
                0
            )
            
            # 使用ewm计算K、D值
            indicators['K'] = pd.Series(indicators['RSV']).ewm(alpha=1/M1).mean()
            indicators['D'] = pd.Series(indicators['K']).ewm(alpha=1/M2).mean()
            indicators['K1'] = indicators['K']  # K1用于背离计算
            
            # 计算多空信号
            indicators['多1'] = np.where(
                indicators['K'] > indicators['K'].shift(1),
                indicators['K'],
                np.nan
            )
            
            indicators['空1'] = np.where(
                indicators['K'] < indicators['K'].shift(1),
                indicators['K'],
                np.nan
            )
            
            indicators['多2'] = np.where(
                indicators['D'] > indicators['D'].shift(1),
                indicators['D'],
                np.nan
            )
            
            indicators['空2'] = np.where(
                indicators['D'] < indicators['D'].shift(1),
                indicators['D'],
                np.nan
            )
            
            # 计算金叉信号
            indicators['BOT_GOLDENCROSS'] = (
                (indicators['K'].shift(1) < indicators['D'].shift(1)) &
                (indicators['K'] > indicators['D']) &
                (indicators['K'] < 30))
            
            # 计算二次金叉信号
            def count_crosses(k, d, lookback=30):
                crosses = ((k.shift(1) < d.shift(1)) & (k > d)).astype(int)
                return crosses.rolling(window=lookback, min_periods=1).sum()
            
            indicators['TWO_GOLDENCROSS'] = (
                (indicators['K'].shift(1) < indicators['D'].shift(1)) &
                (indicators['K'] > indicators['D']) &
                (indicators['D'] < 30) &
                (count_crosses(indicators['K'], indicators['D']) == 2)
            ).astype(int)
            
        except Exception as e:
            print(f"KDJ计算出错: {e}")
            raise
        
    def calculate_ema(self, series, periods):
        """
        计算指数移动平均线 (EMA)
        
        参数:
            series: pandas Series, 需要计算 EMA 的数据
            periods: int, EMA 的周期
        
        返回:
            pandas Series: EMA 值
        """
        # 使用 pandas 的 ewm 函数计算 EMA
        # span = periods, adjust=False 确保与传统 EMA 计算方法一致
        return series.ewm(span=periods, adjust=False).mean()

    def calculate_macd_signals(self):
        """计算 MACD 相关信号"""
        df = self.df.copy()  # 创建副本避免修改原始数据
        
        # 1. 计算 DIF, DEA, MACD
        df['DIF'] = self.calculate_ema(df['Close'], 9) - self.calculate_ema(df['Close'], 26)
        df['DEA'] = self.calculate_ema(df['DIF'], 12)
        df['MACD'] = (df['DIF'] - df['DEA']) * 2
        
        # 2. 计算零轴相关信号
        df['DIF_prev'] = df['DIF'].shift(1)
        df['DEA_prev'] = df['DEA'].shift(1)
        
        # 零轴下金叉
        df['零轴下金叉'] = np.where(
            (df['DIF_prev'] < df['DEA_prev']) & 
            (df['DIF'] > df['DEA']) & 
            (np.maximum(df['DIF'], df['DEA']) <= 0),
            1, 0
        )
        
        # 零轴上金叉
        df['零轴上金叉'] = np.where(
            (df['DIF_prev'] < df['DEA_prev']) & 
            (df['DIF'] > df['DEA']) & 
            (np.minimum(df['DIF'], 0) >= 0),
            1, 0
        )
        
        # 零轴上死叉
        df['零轴上死叉'] = np.where(
            (df['DIF_prev'] > df['DEA_prev']) & 
            (df['DIF'] < df['DEA']) & 
            (np.minimum(df['DIF'], 0) >= 0),
            1, 0
        )
        
        # 零轴下死叉
        df['零轴下死叉'] = np.where(
            (df['DIF_prev'] > df['DEA_prev']) & 
            (df['DIF'] < df['DEA']) & 
            (np.maximum(df['DIF'], df['DEA']) <= 0),
            1, 0
        )
        
        # 3. 计算先机信号
        # V1:=(C*2+H+L)/410
        df['V1'] = (df['Close'] * 2 + df['High'] + df['Low']) / 410
        # V2:=EMA(V1,5)-EMA(V1,34)
        df['V2'] = self.calculate_ema(df['V1'], 5) - self.calculate_ema(df['V1'], 34)
        # V3:=EMA(V2,5)
        df['V3'] = self.calculate_ema(df['V2'], 5)
        # V4:=2*(V2-V3)*5.5/100
        df['V4'] = 2 * (df['V2'] - df['V3']) * 5.5 / 100
        
        # 计算MA60
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 先机:DRAWICON(CROSS(V4,0) AND CLOSE<MA(CLOSE,60) ,0,11)
        df['V4_prev'] = df['V4'].shift(1)
        df['先机信号'] = np.where(
            (df['V4_prev'] < 0) & 
            (df['V4'] > 0) & 
            (df['Close'] < df['MA60']),
            1, 0
        )
        
        # 4. 计算底背离和顶背离信号
        df['EMAMACD'] = self.calculate_ema(df['MACD'], 5)
        df['MACD_CROSS_UP'] = (df['MACD'].shift(1) < df['EMAMACD'].shift(1)) & (df['MACD'] > df['EMAMACD'])
        df['MACD_CROSS_DOWN'] = (df['MACD'].shift(1) > df['EMAMACD'].shift(1)) & (df['MACD'] < df['EMAMACD'])
        
        # 5. 初始化底背离和顶背离列
        df['底背离'] = 0
        df['顶背离'] = 0
        
        # 6. 计算底背离和顶背离
        for i in range(1, len(df)):
            if df['MACD_CROSS_UP'].iloc[i]:
                for j in range(i-1, -1, -1):
                    if df['MACD_CROSS_UP'].iloc[j]:
                        if (df['Close'].iloc[j] > df['Close'].iloc[i] and 
                            df['MACD'].iloc[i] > df['MACD'].iloc[j]):
                            df.loc[df.index[i], '底背离'] = 1
                        break
            
            if df['MACD_CROSS_DOWN'].iloc[i]:
                for j in range(i-1, -1, -1):
                    if df['MACD_CROSS_DOWN'].iloc[j]:
                        if (df['Close'].iloc[j] < df['Close'].iloc[i] and 
                            df['MACD'].iloc[j] > df['MACD'].iloc[i]):
                            df.loc[df.index[i], '顶背离'] = 1
                        break
        
        return df

    def calculate_phantom_force(self):
        """计算幻影主力指标"""
        
        # 1. 海底捞月部分
        def calculate_haidilaoyue(HIGH, LOW, OPEN, CLOSE):
            VAR1 = REF((LOW + OPEN + CLOSE + HIGH) / 4, 1)
            
            # 多头部分
            VAR2 = SMA(ABS(LOW - VAR1), 13, 1) / SMA(MAX(LOW - VAR1, 0), 10, 1)
            VAR3 = EMA(VAR2, 10)
            VAR4 = LLV(LOW, 33)
            VAR5 = EMA(IF(LOW <= VAR4, VAR3, 0), 3)
            VAR6 = POW(VAR5, 0.3)
            
            # 空头部分
            VAR21 = SMA(ABS(HIGH - VAR1), 13, 1) / SMA(MIN(HIGH - VAR1, 0), 10, 1)
            VAR31 = EMA(VAR21, 10)
            VAR41 = HHV(HIGH, 33)
            VAR51 = EMA(IF(HIGH >= VAR41, -VAR31, 0), 3)
            VAR61 = POW(VAR51, 0.3)
            
            # 缩放比例
            RADIO1 = 200 / CONST(HHV(MAX(VAR6, VAR61), len(HIGH)))
            
            # 计算最终结果
            BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
            LIRED = IF(VAR51 > REF(VAR51, 1), -VAR61 * RADIO1, 0)
            
            return BLUE, LIRED
        
        # 2. 资金力度部分
        def calculate_zijinlidu(HIGH, LOW, OPEN, CLOSE, VOL):
            QJJ = VOL / ((HIGH - LOW) * 2 - ABS(CLOSE - OPEN))
            XVL = IF(CLOSE == OPEN, 0, (CLOSE - OPEN) * QJJ)
            HSL = XVL / 20 / 1.15
            attack_flow = HSL * 0.55 + REF(HSL, 1) * 0.33 + REF(HSL, 2) * 0.22
            LLJX = EMA(attack_flow, 3)
            
            # 缩放比例
            RADIO = 10000 / CONST(HHV(VOL, len(VOL)))
            
            RED = IF(LLJX > 0, LLJX * RADIO, 0)
            YELLOW = IF(HSL > 0, HSL * 0.6 * RADIO, 0)
            GREEN = IF((LLJX < 0) | (HSL < 0), MIN(LLJX, HSL * 0.6) * RADIO, 0)
            LIGHTBLUE = DMA(attack_flow, 0.222228) * RADIO
            
            return RED, YELLOW, GREEN, LIGHTBLUE, attack_flow
        
        # 3. KDJ部分
        def calculate_kdj_custom(HIGH, LOW, CLOSE):
            RSV1 = (CLOSE - LLV(LOW, 39)) / (HHV(HIGH, 39) - LLV(LOW, 39)) * 100
            K = SMA(RSV1, 2, 1)
            D = SMA(K, 2, 1)
            J = 3 * K - 2 * D
            PINK = SMA(J, 2, 1)
            
            return PINK
        
        # 计算所有指标
        HIGH, LOW = self.df['High'].values, self.df['Low'].values
        OPEN, CLOSE = self.df['Open'].values, self.df['Close'].values
        VOL = self.df['Volume'].values
        
        # 计算三个部分的指标
        BLUE, LIRED = calculate_haidilaoyue(HIGH, LOW, OPEN, CLOSE)
        RED, YELLOW, GREEN, LIGHTBLUE, attack_flow = calculate_zijinlidu(HIGH, LOW, OPEN, CLOSE, VOL)
        PINK = calculate_kdj_custom(HIGH, LOW, CLOSE)
        
        # 生成信号
        buy_signals = CROSS(10, PINK)  # PINK线上穿10
        sell_signals = CROSS(PINK, 94)  # PINK线下穿90
        
        # 保存结果到DataFrame
        self.df['phantom_blue'] = BLUE
        self.df['phantom_red'] = RED
        self.df['phantom_yellow'] = YELLOW
        self.df['phantom_green'] = GREEN
        self.df['phantom_pink'] = PINK
        self.df['phantom_lightblue'] = LIGHTBLUE
        self.df['phantom_buy'] = buy_signals
        self.df['phantom_sell'] = sell_signals
        
        return self.df  

    def calculate_safe_zone(self):
        """计算安全区域指标"""
        df = self.df
        CLOSE, HIGH, LOW = df['Close'].values, df['High'].values, df['Low'].values
        
        # 计算顶底线
        RSVA1 = (CLOSE - LLV(LOW, 9)) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100
        RSVA2 = 100 * (HHV(HIGH, 9) - CLOSE) / (HHV(HIGH, 9) - LLV(LOW, 9))
        VAR21 = SMA(RSVA2, 9, 1) + 100
        VAR11 = SMA(RSVA1, 3, 1)
        VAR51 = SMA(VAR11, 3, 1) + 100
        df['顶底线'] = VAR51 - VAR21 + 50
        
        # 计算趋势
        VAR2 = LLV(LOW, 330)
        VAR3 = HHV(HIGH, 210)
        VAR4 = EMA((CLOSE - VAR2) / (VAR3 - VAR2) * 100, 10) * -1 + 100
        df['趋势'] = 100 - EMA(0.191 * REF(VAR4, 1) + 0.809 * VAR4, 1)
        
        # 计算区域信号
        df['顶底线上升'] = df['顶底线'] > REF(df['顶底线'].values, 1)
        df['顶底线下降'] = df['顶底线'] < REF(df['顶底线'].values, 1)
        df['趋势上升'] = df['趋势'] > REF(df['趋势'].values, 1)
        df['趋势下降'] = df['趋势'] < REF(df['趋势'].values, 1)
        
        # 计算强拉升信号
        Y1 = LLV(LOW, 17)
        Y2 = SMA(ABS(LOW - REF(LOW, 1)), 17, 1)
        Y3 = SMA(MAX(LOW - REF(LOW, 1), 0), 17, 2)
        Q = -(EMA(IF(LOW <= Y1, Y2/Y3, -3), 1))
        df['强拉升'] = CROSS(Q, 0)
        
        # 计算加强拉升信号
        Q1 = (CLOSE - MA(CLOSE, 40)) / MA(CLOSE, 40) * 100
        df['加强拉升'] = CROSS(Q1, -24)
        
        # 计算买半注信号
        VAR22_temp = (2 * CLOSE + HIGH + LOW) / 4
        VAR22 = EMA(EMA(EMA(VAR22_temp, 4), 4), 4)  # 使用EMA替代EXPMA
        天 = MA((VAR22 - REF(VAR22, 1)) / REF(VAR22, 1) * 100, 2)
        地 = MA((VAR22 - REF(VAR22, 1)) / REF(VAR22, 1) * 100, 1)
        df['买半注'] = (地 > 天) & (地 < 0)
        
        # 计算TREND1
        VAR1B = (HHV(HIGH, 9) - CLOSE) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100 - 70
        VAR2B = SMA(VAR1B, 9, 1) + 100
        VAR3B = (CLOSE - LLV(LOW, 9)) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100
        VAR4B = SMA(VAR3B, 3, 1)
        VAR5B = SMA(VAR4B, 3, 1) + 100
        VAR6B = VAR5B - VAR2B
        df['TREND1'] = IF(VAR6B > 45, VAR6B - 45, 0)
        df['TREND1上升'] = REF(df['TREND1'], 1) < df['TREND1']
        df['TREND1下降'] = REF(df['TREND1'], 1) > df['TREND1']
        
        # 计算火焰山
        VAR2Q = REF(LOW, 1)
        VAR3Q = SMA(ABS(LOW - VAR2Q), 3, 1) / SMA(MAX(LOW - VAR2Q, 0), 3, 1) * 100
        VAR4Q = EMA(IF(CLOSE * 1.3, VAR3Q * 10, VAR3Q / 10), 3)
        VAR5Q = LLV(LOW, 30)
        VAR6Q = HHV(VAR4Q, 30)
        VAR7Q = IF(MA(CLOSE, 58), 1, 0)
        VAR8Q = EMA(IF(LOW <= VAR5Q, (VAR4Q + VAR6Q * 2) / 2, 0), 3) / 999 * VAR7Q
        df['火焰山'] = IF(VAR8Q > 100, 100, VAR8Q)
        
        # 计算BBUY信号
        D1 = (CLOSE + LOW + HIGH) / 3
        D2 = EMA(D1, 6)
        D3 = EMA(D2, 5)
        df['BBUY'] = CROSS(D2, D3)
        
        # 计算减仓信号
        VARR1 = SMA(MAX(CLOSE - REF(CLOSE, 1), 0), 6, 1) / SMA(ABS(CLOSE - REF(CLOSE, 1)), 6, 1) * 100
        df['减仓'] = CROSS(80, VARR1)
        
        # 计算趋势和能量线
        V1 = (CLOSE - LLV(LOW, 25)) / (HHV(HIGH, 25) - LLV(LOW, 25)) * 100
        V2 = SMA(V1, 3, 1)
        TREND = SMA(V2, 3, 1)
        POWERLINE = SMA(TREND, 3, 1)
        
        # 计算买卖信号
        df['BUY'] = IF(CROSS(TREND, POWERLINE) & (TREND < 25), 20, 0)
        df['SOLD'] = IF(CROSS(POWERLINE, TREND) & (POWERLINE > 80), 85, 100)
        
        # 计算区域
        df['高安全区'] = (df['趋势'] >= 90)
        df['安全区'] = (df['趋势'] >= 80) & (df['趋势'] < 90)
        df['粉区持币'] = (df['趋势'] >= 45) & (df['趋势'] < 80)
        df['绿区持股'] = (df['趋势'] >= 20) & (df['趋势'] < 45)
        df['风险区'] = (df['趋势'] >= 10) & (df['趋势'] < 20)
        df['高风险区'] = (df['趋势'] < 10)
        
        return df


# 示例代码
if __name__ == "__main__":
    # 读取股票数据，例如 AAPL 股票数据
    df = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)
    # 创建 StockAnalysis 类的实例
    analysis = StockAnalysis(df)
    # 计算所有策略
    df = analysis.calculate_all_indicators()
    # 打印或保存最终的 DataFrame
    print(df.head())