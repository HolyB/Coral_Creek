import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_utils import prepare_data, evaluation_metric
from xgboost import XGBRegressor
from stock_data_fetcher import StockDataFetcher
from arima_forecast import ARIMAStockForecast

class XGBoostStockForecast:
    def __init__(self, df, ticker):
        self.df = df
        self.ticker = ticker
        self.train_set = None
        self.test_set = None
        self.train_end_date = None
        self.test_start_date = None
        self.residuals = None
        self.y = None
        self.yhat = None

    def preprocess_data(self, train_end_date, test_start_date):
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.df.index = pd.to_datetime(self.df.index)
        
        # Split the data into train and test sets based on dates
        self.train_set = self.df[:self.train_end_date]
        self.test_set = self.df[self.test_start_date:]
        print("Preprocessing completed: Train set and Test set split.")
        print(f"Train set length: {len(self.train_set)}, Test set length: {len(self.test_set)}")

    def perform_diff(self):
        self.df['diff_1'] = self.df['close'].diff(1)
        self.df['diff_2'] = self.df['diff_1'].diff(1)
        self.train_set.loc[:, 'diff_1'] = self.train_set['close'].diff(1)
        self.train_set.loc[:, 'diff_2'] = self.train_set['diff_1'].diff(1)
        self.test_set.loc[:, 'diff_1'] = self.test_set['close'].diff(1)
        self.test_set.loc[:, 'diff_2'] = self.test_set['diff_1'].diff(1)
        print("Differencing completed.")

    def fit_xgboost_model(self):
        train, test = prepare_data(self.train_set, n_test=len(self.test_set), n_in=6, n_out=1)
        print("Prepared data for XGBoost:")
        print(f"Train set shape: {train.shape}, Test set shape: {test.shape}")
        if train.empty or test.empty:
            print("Error: Train or test set is empty after prepare_data.")
            return None, None
        self.y, self.yhat = self.walk_forward_validation(train, test)
        if not self.yhat:
            print("Error: No predictions were generated.")
            return None, None
        return self.y, self.yhat

    def walk_forward_validation(self, train, test):
        predictions = list()
        train = train.values
        history = [x for x in train]
        for i in range(len(test)):
            testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
            print(f"Iteration {i}: TestX shape: {testX.shape}, Testy: {testy}")
            if len(history) == 0 or testX.isnull().any():
                print(f"Skipping iteration {i} due to empty history or null values in testX")
                continue
            yhat = self.xgboost_forecast(history, testX)
            if not np.isnan(yhat):
                predictions.append(yhat)
                history.append(test.iloc[i, :])
            else:
                print(f"Iteration {i}: Prediction was NaN.")
        return test.iloc[:, -1], predictions

    # Helper function to check for infinities or large values
    def check_for_inf_or_large_values(self, data, label):
        try:
            data = data.astype(np.float64)  # Ensure data is in a numeric type that supports inf checks
            if np.isinf(data).any():
                print(f"Error: {label} contains infinity values.")
            if (data > 1e308).any():
                print(f"Error: {label} contains values too large.")
            if (data < -1e308).any():
                print(f"Error: {label} contains values too small.")
            if np.isnan(data).any():
                print(f"Error: {label} contains NaN values.")
        except ValueError as e:
            print(f"Error: {label} cannot be checked for infinities or large values due to type issues: {e}")

    def xgboost_forecast(self, train, testX):
        train = np.asarray(train)
        if train.shape[0] == 0 or testX.shape[0] == 0:
            print("Error: Train or test data is empty for XGBoost forecast.")
            return np.nan
        trainX, trainy = train[:, :-1], train[:, -1]
        print(f"Training XGBoost model: TrainX shape: {trainX.shape}, Trainy shape: {trainy.shape}")
        model = XGBRegressor(objective='reg:squarederror', n_estimators=20)
        self.check_for_inf_or_large_values(trainX, "trainX")
        model.fit(trainX, trainy)
        yhat = model.predict(np.asarray([testX]))
        print(f"Prediction for current step: {yhat[0]}")
        return yhat[0]

    def plot_residuals(self):
        if self.y is not None and len(self.y) == len(self.test_set.index):
            plt.figure(figsize=(10, 6))
            plt.plot(self.test_set.index, self.y, label='Residuals')
            plt.plot(self.test_set.index, self.yhat, label='Predicted Residuals')
            plt.title(f'ARIMA+XGBoost: Residuals Prediction for {self.ticker}')
            plt.xlabel('Time', fontsize=12, verticalalignment='top')
            plt.ylabel('Residuals', fontsize=14, horizontalalignment='center')
            plt.legend()
            plt.show()
        else:
            print("Mismatch between test set index length and residuals length")

    def evaluate_final_predictions(self, arima_predictions):
        if self.yhat is not None and len(arima_predictions) >= len(self.yhat):
                  # Select the 'close' column and convert to float
            arima_predictions_close = arima_predictions['close'].astype(float)  
            # Perform the addition for the 'close' column values
            finalpredicted_stock_price = [i + j for i, j in zip(arima_predictions_close[-len(self.yhat):], self.yhat)] 

            # Evaluate model performance
            actual_prices = self.test_set['close']
            if len(actual_prices) != len(finalpredicted_stock_price):
                print("Mismatch between actual prices and predicted prices length")
                return
            evaluation_metric(actual_prices, finalpredicted_stock_price)

            # Plot actual vs predicted stock prices
            plt.figure(figsize=(10, 6))
            plt.plot(self.test_set.index, actual_prices, label='Stock Price')
            plt.plot(self.test_set.index, finalpredicted_stock_price, label='Predicted Stock Price')
            plt.title(f'ARIMA+XGBoost: Stock Price Prediction for {self.ticker}')
            plt.xlabel('Time', fontsize=12, verticalalignment='top')
            plt.ylabel('Close', fontsize=14, horizontalalignment='center')
            plt.legend()
            plt.show()
        else:
            print("Insufficient ARIMA predictions for evaluation")

# Example usage
if __name__ == "__main__":
    # Load dataset
    fetcher = StockDataFetcher('CELH', source='yahoo', interval='1d')
    data = fetcher.get_stock_data()
    print(f"Data for CELH:\n", data.head())

    # Initialize ARIMAStockForecast to get ARIMA predictions
    arima_forecast = ARIMAStockForecast(data, ticker='CELH')
    arima_forecast.preprocess_data(train_end_date='2024-08-01', test_start_date='2024-08-02')
    model_fit = arima_forecast.fit_arima_model(order=(2, 1, 0))
    arima_predictions = arima_forecast.forecast(model_fit)['close'].values

    # Initialize XGBoostStockForecast
    xgboost_forecast = XGBoostStockForecast(data, ticker='CELH')

    # Preprocess data
    xgboost_forecast.preprocess_data(train_end_date='2024-08-01', test_start_date='2024-08-02')

    # Perform differencing
    xgboost_forecast.perform_diff()

    # Fit XGBoost model
    y, yhat = xgboost_forecast.fit_xgboost_model()

    # Plot residuals
    if y is not None and yhat is not None:
        xgboost_forecast.plot_residuals()

    # Evaluate final predictions
    if yhat is not None:
        xgboost_forecast.evaluate_final_predictions(arima_predictions)