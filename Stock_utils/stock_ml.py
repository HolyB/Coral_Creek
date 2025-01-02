import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# 使用模块中的类
from stock_analysis import StockAnalysis
from label_generation import LabelGenerator
from stock_data_fetcher import StockDataFetcher
from notification import SignalNotifier

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取配置文件
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 从配置文件中获取tickers
tickers = config['tickers']
# 定义时间周期
intervals = ['1d']

# 加载特征配置文件
feature_config_path = os.path.join(current_dir, 'feature_config.yaml')
with open(feature_config_path, 'r', encoding='utf-8') as file:
    feature_config = yaml.safe_load(file)

categorical_features = feature_config['categorical_features']
numerical_features = feature_config['numerical_features']

# 创建一个机器学习类来预测涨跌概率和买点概率
class StockMLModel:
    def __init__(self, data):
        self.data = data
        self.model_up_down = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_buy_point = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # 特征和标签
        features = self.data[numerical_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        labels_up_down = np.where(self.data['pct_change_tomorrow'] > 0, 1, 0)  # 涨跌标签
        labels_buy_point = self.data['is_lowest_7days'].astype(int)  # 买点标签
        
        # 特征缩放
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, labels_up_down, labels_buy_point

    def train_models(self):
        features, labels_up_down, labels_buy_point = self.preprocess_data()
        
        # 训练涨跌模型
        X_train, X_test, y_train, y_test = train_test_split(features, labels_up_down, test_size=0.2, random_state=42)
        self.model_up_down.fit(X_train, y_train)
        
        # 训练买点模型
        X_train, X_test, y_train, y_test = train_test_split(features, labels_buy_point, test_size=0.2, random_state=42)
        self.model_buy_point.fit(X_train, y_train)

    def predict(self, new_data):
        new_data_scaled = self.scaler.transform(new_data.replace([np.inf, -np.inf], np.nan).fillna(0))
        up_down_prob = self.model_up_down.predict_proba(new_data_scaled)[:, 1]  # 涨跌的概率
        buy_point_prob = self.model_buy_point.predict_proba(new_data_scaled)[:, 1]  # 买点的概率
        
        return up_down_prob, buy_point_prob