import pandas as pd

class LabelGenerator:
    def __init__(self, df):
        self.df = df.copy()
        self.rename_columns()

    def rename_columns(self):
        self.df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)

    def pct_change_today(self):
        self.df['pct_change_today'] = (self.df['close'] - self.df['open']) / self.df['open'] * 100
        return self.df['pct_change_today']

    def pct_change_tomorrow(self):
        self.df['pct_change_tomorrow'] = (self.df['close'].shift(-1) - self.df['open']) / self.df['open'] * 100
        return self.df['pct_change_tomorrow']

    def pct_change_7days(self):
        self.df['pct_change_7days'] = (self.df['close'].shift(-7) - self.df['open']) / self.df['open'] * 100
        return self.df['pct_change_7days']

    def pct_change_15days(self):
        self.df['pct_change_15days'] = (self.df['close'].shift(-15) - self.df['open']) / self.df['open'] * 100
        return self.df['pct_change_15days']

    def pct_change_30days(self):
        self.df['pct_change_30days'] = (self.df['close'].shift(-30) - self.df['open']) / self.df['open'] * 100
        return self.df['pct_change_30days']

    def is_lowest_7days(self):
        self.df['is_lowest_7days'] = self.df['close'] == self.df['close'].rolling(window=7).min()
        return self.df['is_lowest_7days']

    def is_lowest_30days(self):
        self.df['is_lowest_30days'] = self.df['close'] == self.df['close'].rolling(window=30).min()
        return self.df['is_lowest_30days']

    def is_highest_7days(self):
        self.df['is_highest_7days'] = self.df['close'] == self.df['close'].rolling(window=7).max()
        return self.df['is_highest_7days']

    def is_highest_30days(self):
        self.df['is_highest_30days'] = self.df['close'] == self.df['close'].rolling(window=30).max()
        return self.df['is_highest_30days']

    def generate_labels(self):
        self.pct_change_today()
        self.pct_change_tomorrow()
        self.pct_change_7days()
        self.pct_change_15days()
        self.pct_change_30days()
        self.is_lowest_7days()
        self.is_lowest_30days()
        self.is_highest_7days()
        self.is_highest_30days()
        return self.df

# Example usage
if __name__ == "__main__":
    # Sample dataframe with OHLCV data
    data = {
        'Open': [100, 102, 101, 105, 110, 112, 115],
        'Close': [102, 101, 104, 110, 115, 118, 117],
        'High': [103, 105, 107, 111, 116, 119, 118],
        'Low': [99, 100, 100, 103, 109, 111, 114],
        'Adj Close': [102, 101, 104, 110, 115, 118, 117],
        'Volume': [1000, 1200, 1300, 1500, 1400, 1700, 1800]
    }
    df = pd.DataFrame(data)
    
    # Generate labels
    label_generator = LabelGenerator(df)
    df_with_labels = label_generator.generate_labels()
    
    # Display the result
    print(df_with_labels)