#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Learning Models - 深度学习模型

LSTM/GRU 时间序列预测模型
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 添加父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def check_torch_available() -> bool:
    """检查 PyTorch 是否可用"""
    return TORCH_AVAILABLE


class StockDataset(Dataset):
    """股票时序数据集"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM 价格预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out


class GRUModel(nn.Module):
    """GRU 价格预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.fc(last_output)
        return out


def prepare_sequence_data(df: pd.DataFrame, seq_length: int = 20, 
                          target_col: str = 'Close', 
                          feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备时序训练数据
    
    Args:
        df: 包含 OHLCV 的 DataFrame
        seq_length: 序列长度 (回看天数)
        target_col: 目标列
        feature_cols: 特征列列表
    
    Returns:
        (X, y): 训练数据
    """
    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 归一化
    data = df[feature_cols].values
    target = df[target_col].values
    
    # Min-Max 归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    data_scaled = scaler_X.fit_transform(data)
    target_scaled = scaler_y.fit_transform(target.reshape(-1, 1))
    
    # 创建序列
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length])
        # 预测下一天的收盘价
        y.append(target_scaled[i + seq_length])
    
    return np.array(X), np.array(y), scaler_y


class DeepLearningTrainer:
    """深度学习训练器"""
    
    def __init__(self, model_type: str = 'LSTM', 
                 hidden_size: int = 64, num_layers: int = 2):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler_y = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self, input_size: int):
        """构建模型"""
        if self.model_type == 'LSTM':
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
        else:  # GRU
            self.model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
        
        self.model.to(self.device)
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 0.001) -> Dict:
        """
        训练模型
        
        Returns:
            训练历史和指标
        """
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        # 数据集
        train_dataset = StockDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = StockDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        self.history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # 验证
            if X_val is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                self.history['val_loss'].append(avg_val_loss)
        
        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'epochs': epochs,
            'model_type': self.model_type
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def get_training_chart_data(self) -> Dict:
        """获取训练曲线数据"""
        return {
            'epochs': list(range(1, len(self.history['train_loss']) + 1)),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss']
        }


def train_price_predictor(symbol: str, days: int = 100, 
                          seq_length: int = 20, epochs: int = 50,
                          model_type: str = 'LSTM') -> Dict:
    """
    训练价格预测器
    
    Args:
        symbol: 股票代码
        days: 使用多少天的历史数据
        seq_length: 序列长度
        epochs: 训练轮数
        model_type: 'LSTM' 或 'GRU'
    
    Returns:
        训练结果
    """
    from data_fetcher import get_us_stock_data
    from sklearn.model_selection import train_test_split
    
    # 获取数据
    df = get_us_stock_data(symbol, days=days)
    if df is None or len(df) < seq_length + 10:
        return {'error': f'Insufficient data for {symbol}'}
    
    # 准备序列数据
    X, y, scaler = prepare_sequence_data(df, seq_length=seq_length)
    
    if len(X) < 20:
        return {'error': 'Not enough sequences'}
    
    # 分割训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 训练
    trainer = DeepLearningTrainer(model_type=model_type)
    trainer.build_model(X.shape[2])
    trainer.scaler_y = scaler
    
    metrics = trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    # 预测最后一段
    predictions = trainer.predict(X_val)
    predictions_rescaled = scaler.inverse_transform(predictions)
    actuals_rescaled = scaler.inverse_transform(y_val)
    
    # 计算指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(actuals_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(actuals_rescaled, predictions_rescaled))
    
    # 方向准确率
    direction_correct = np.sum(
        (predictions_rescaled[1:] - predictions_rescaled[:-1]) * 
        (actuals_rescaled[1:] - actuals_rescaled[:-1]) > 0
    )
    direction_accuracy = direction_correct / (len(predictions_rescaled) - 1) if len(predictions_rescaled) > 1 else 0
    
    return {
        'symbol': symbol,
        'model_type': model_type,
        'epochs': epochs,
        'mae': float(mae),
        'rmse': float(rmse),
        'direction_accuracy': float(direction_accuracy),
        'train_loss': metrics['final_train_loss'],
        'val_loss': metrics['final_val_loss'],
        'chart_data': trainer.get_training_chart_data(),
        'predictions': predictions_rescaled.flatten().tolist()[-10:],
        'actuals': actuals_rescaled.flatten().tolist()[-10:]
    }


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not installed. Run: pip install torch")
    else:
        print("✅ PyTorch available")
        print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # 测试训练
        result = train_price_predictor('AAPL', days=100, epochs=20)
        print(f"\n🧠 Training Result:")
        print(f"   MAE: ${result.get('mae', 0):.2f}")
        print(f"   RMSE: ${result.get('rmse', 0):.2f}")
        print(f"   Direction Accuracy: {result.get('direction_accuracy', 0):.1%}")
