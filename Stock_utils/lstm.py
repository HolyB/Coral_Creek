import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from model_utils import evaluation_metric

class LSTMStockForecast:
    def __init__(self, df, ticker):
        self.df = df
        self.ticker = ticker
        self.train_set = None
        self.test_set = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.latest_predictions = None

    def preprocess_data(self, train_end_date, test_start_date):
        # Convert dates to datetime
        train_end_date = pd.to_datetime(train_end_date)
        test_start_date = pd.to_datetime(test_start_date)
        self.df.index = pd.to_datetime(self.df.index)
        
        # Split the data into train and test sets
        self.train_set = self.df[:train_end_date]
        self.test_set = self.df[test_start_date:]
        
        # Fill NaN values with -9999
        self.train_set = self.train_set.apply(lambda x: x.fillna(-9999), axis=0)
        self.test_set = self.test_set.apply(lambda x: x.fillna(-9999), axis=0)
        
        # Scale the data
        training_set_scaled = self.scaler.fit_transform(self.train_set)
        testing_set_scaled = self.scaler.transform(self.test_set)

        # Prepare training and testing data
        self.train_X, self.train_Y = self.create_dataset(training_set_scaled, time_step=1)
        self.test_X, self.test_Y = self.create_dataset(testing_set_scaled, time_step=1)

        # Reshape input to be [samples, time steps, features]
        self.train_X = self.train_X.reshape(self.train_X.shape[0], self.train_X.shape[1], self.df.shape[1])
        self.test_X = self.test_X.reshape(self.test_X.shape[0], self.test_X.shape[1], self.df.shape[1])

    def create_dataset(self, dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), :])
            Y.append(dataset[i + time_step, self.df.columns.get_loc('close')])  # Use 'close' column for Y
        return np.array(X), np.array(Y)

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        adam = Adam(learning_rate=0.001)
        self.model.compile(optimizer=adam, loss='mse')

    def train_model(self, epochs=100, batch_size=32):
        self.model.fit(self.train_X, self.train_Y, epochs=epochs, batch_size=batch_size, validation_data=(self.test_X, self.test_Y), verbose=1)

    def evaluate_model(self):
        # Make predictions
        predicted_stock_price = self.model.predict(self.test_X)
        predicted_stock_price = self.scaler.inverse_transform(np.concatenate((self.test_X[:, :, :-1].reshape(self.test_X.shape[0], -1), predicted_stock_price), axis=1))[:, self.df.columns.get_loc('close')]
        real_stock_price = self.scaler.inverse_transform(np.concatenate((self.test_X[:, :, :-1].reshape(self.test_X.shape[0], -1), self.test_Y.reshape(-1, 1)), axis=1))[:, self.df.columns.get_loc('close')]

        # Evaluation
        evaluation_metric(real_stock_price, predicted_stock_price)

        # Plot actual vs predicted stock prices
        plt.figure(figsize=(10, 6))
        plt.plot(real_stock_price, label='Actual Stock Price')
        plt.plot(predicted_stock_price, label='Predicted Stock Price')
        plt.title(f'LSTM: Stock Price Prediction for {self.ticker}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def get_latest_prediction(self):
        # Make prediction for all the test data points
        latest_predictions = self.model.predict(self.test_X)
        self.latest_predictions = self.scaler.inverse_transform(np.concatenate((self.test_X[:, :, :-1].reshape(self.test_X.shape[0], -1), latest_predictions), axis=1))[:, self.df.columns.get_loc('close')]
        print(self.latest_predictions)
        return self.latest_predictions

# Example usage
if __name__ == "__main__":
    # Load dataset
    df_with_selected_features = pd.read_csv('your_dataset.csv', index_col='trade_date', parse_dates=True)

    # Initialize LSTMStockForecast
    lstm_forecast = LSTMStockForecast(df_with_selected_features, ticker='CELH')

    # Preprocess data
    lstm_forecast.preprocess_data(train_end_date='2024-08-01', test_start_date='2024-08-02')

    # Build and train the model
    lstm_forecast.build_model()
    lstm_forecast.train_model()

    # Evaluate the model 
    lstm_forecast.evaluate_model()

    # Get the latest prediction
    latest_predictions = lstm_forecast.get_latest_prediction()
    print(f"Latest Predictions: {latest_predictions}")